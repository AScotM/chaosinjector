#!/usr/bin/env python3

import argparse
import asyncio
import gc
import json
import logging
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Deque, Dict, Iterable, List, Optional, Sequence

import psutil


logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
MAX_EVENTS_HISTORY = 1000
MAX_CONCURRENT_SPIKES = 3
MAX_CONCURRENT_INJECTIONS = 5
DEFAULT_EXPERIMENT_DURATION = 10
DEFAULT_MIN_EXPERIMENT_INTERVAL = 10.0
DEFAULT_MAX_EXPERIMENT_INTERVAL = 30.0


class ChaosStrategy(Enum):
    LATENCY = "latency"
    FAILURE = "failure"
    GATEWAY_OUTAGE = "gateway_outage"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    NETWORK_PARTITION = "network_partition"
    RANDOM = "random"


class ChaosIntensity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


INTENSITY_MULTIPLIERS = {
    ChaosIntensity.LOW: 0.5,
    ChaosIntensity.MEDIUM: 1.0,
    ChaosIntensity.HIGH: 2.0,
    ChaosIntensity.EXTREME: 4.0,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_event_id(prefix: str) -> str:
    timestamp = time.time_ns()
    random_part = random.SystemRandom().randrange(1000, 10000)
    return f"{prefix}_{timestamp}_{random_part}"


@dataclass(frozen=True)
class ChaosConfig:
    latency_probability: float = 0.03
    failure_probability: float = 0.02
    gateway_outage_probability: float = 0.02
    memory_leak_probability: float = 0.01
    cpu_spike_probability: float = 0.01
    network_partition_probability: float = 0.005

    max_latency: float = 2.0
    max_memory_leak_mb: int = 100
    max_cpu_spike_duration: float = 5.0

    gateway_outage_duration: float = 30.0
    network_partition_duration: float = 60.0

    enabled_strategies: tuple[ChaosStrategy, ...] = field(
        default_factory=lambda: (
            ChaosStrategy.LATENCY,
            ChaosStrategy.FAILURE,
            ChaosStrategy.GATEWAY_OUTAGE,
        )
    )

    intensity: ChaosIntensity = ChaosIntensity.MEDIUM

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        probabilities = {
            "latency_probability": self.latency_probability,
            "failure_probability": self.failure_probability,
            "gateway_outage_probability": self.gateway_outage_probability,
            "memory_leak_probability": self.memory_leak_probability,
            "cpu_spike_probability": self.cpu_spike_probability,
            "network_partition_probability": self.network_partition_probability,
        }

        for name, value in probabilities.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        positive_values = {
            "max_latency": self.max_latency,
            "max_memory_leak_mb": self.max_memory_leak_mb,
            "max_cpu_spike_duration": self.max_cpu_spike_duration,
            "gateway_outage_duration": self.gateway_outage_duration,
            "network_partition_duration": self.network_partition_duration,
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if not isinstance(self.intensity, ChaosIntensity):
            raise TypeError("intensity must be a ChaosIntensity")

        if not self.enabled_strategies:
            raise ValueError("At least one chaos strategy must be enabled")

        for strategy in self.enabled_strategies:
            if not isinstance(strategy, ChaosStrategy):
                raise TypeError(f"Invalid strategy type: {strategy!r}")

    def probability_for(self, strategy: ChaosStrategy) -> float:
        base_probability = {
            ChaosStrategy.LATENCY: self.latency_probability,
            ChaosStrategy.FAILURE: self.failure_probability,
            ChaosStrategy.GATEWAY_OUTAGE: self.gateway_outage_probability,
            ChaosStrategy.MEMORY_LEAK: self.memory_leak_probability,
            ChaosStrategy.CPU_SPIKE: self.cpu_spike_probability,
            ChaosStrategy.NETWORK_PARTITION: self.network_partition_probability,
        }.get(strategy)

        if base_probability is None:
            return 1.0

        multiplier = INTENSITY_MULTIPLIERS[self.intensity]
        return min(base_probability * multiplier, 1.0)

    def with_intensity(self, intensity: ChaosIntensity) -> "ChaosConfig":
        return replace(self, intensity=intensity)


@dataclass(frozen=True)
class ChaosEvent:
    event_id: str
    strategy: ChaosStrategy
    intensity: ChaosIntensity
    timestamp: datetime
    description: str
    affected_component: str
    duration: float = 0.0
    impact: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("event_id cannot be empty")

        normalized = self.event_id.replace("_", "").replace("-", "")
        if not normalized.isalnum():
            raise ValueError("Invalid event_id format")

        if not isinstance(self.strategy, ChaosStrategy):
            raise TypeError("strategy must be a ChaosStrategy")

        if not isinstance(self.intensity, ChaosIntensity):
            raise TypeError("intensity must be a ChaosIntensity")

        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")

        if not self.affected_component:
            raise ValueError("affected_component cannot be empty")

        if self.duration < 0:
            raise ValueError("duration cannot be negative")

        if len(self.description) > 500:
            raise ValueError("Description too long")

        if len(self.impact) > 500:
            raise ValueError("Impact too long")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "strategy": self.strategy.value,
            "intensity": self.intensity.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "affected_component": self.affected_component,
            "duration": self.duration,
            "impact": self.impact,
        }


class ChaosMonitor:
    def __init__(self, max_events: int = MAX_EVENTS_HISTORY):
        if max_events <= 0:
            raise ValueError("max_events must be positive")

        self._events: Deque[ChaosEvent] = deque(maxlen=max_events)
        self._event_count = 0
        self._start_time = time.monotonic()
        self._strategy_stats: Dict[ChaosStrategy, Dict[str, Any]] = {}

    @property
    def events(self) -> List[ChaosEvent]:
        return list(self._events)

    def record_event(self, event: ChaosEvent) -> None:
        self._events.append(event)
        self._event_count += 1

        stats = self._strategy_stats.setdefault(
            event.strategy,
            {
                "count": 0,
                "total_duration": 0.0,
                "last_occurrence": event.timestamp,
            },
        )

        stats["count"] += 1
        stats["total_duration"] += event.duration
        stats["last_occurrence"] = event.timestamp

        logger.info("Chaos event recorded: %s", event.description)

    def get_metrics(self) -> Dict[str, Any]:
        runtime = max(time.monotonic() - self._start_time, 0.0)

        events_by_strategy = {
            strategy.value: 0
            for strategy in ChaosStrategy
        }

        for event in self._events:
            events_by_strategy[event.strategy.value] += 1

        strategy_details: Dict[str, Any] = {}

        for strategy, stats in self._strategy_stats.items():
            count = stats["count"]

            strategy_details[strategy.value] = {
                "total_events": count,
                "average_duration": (
                    stats["total_duration"] / count
                    if count
                    else 0.0
                ),
                "last_occurrence": stats["last_occurrence"].isoformat(),
            }

        events_per_minute = (
            self._event_count / (runtime / 60.0)
            if runtime > 0
            else 0.0
        )

        return {
            "total_events": self._event_count,
            "retained_events": len(self._events),
            "runtime_seconds": runtime,
            "events_per_minute": events_per_minute,
            "events_by_strategy": events_by_strategy,
            "strategy_details": strategy_details,
            "recent_events": [
                event.to_dict()
                for event in list(self._events)[-10:]
            ],
        }

    def clear_events(self) -> None:
        self._events.clear()
        self._event_count = 0
        self._strategy_stats.clear()
        self._start_time = time.monotonic()


class MemoryLeakSimulator:
    def __init__(self, max_leak_mb: int):
        if max_leak_mb <= 0:
            raise ValueError("max_leak_mb must be positive")

        self.max_leak_mb = max_leak_mb
        self._leaked_data: List[List[bytearray]] = []
        self._total_leaked_mb = 0
        self._lock = asyncio.Lock()

    async def inject_memory_leak(
        self,
        size_mb: int,
        intensity: ChaosIntensity,
    ) -> ChaosEvent:
        if size_mb <= 0:
            raise ValueError("size_mb must be positive")

        async with self._lock:
            available_mb = self.max_leak_mb - self._total_leaked_mb

            if available_mb <= 0:
                return ChaosEvent(
                    event_id=generate_event_id("memleak_skipped"),
                    strategy=ChaosStrategy.MEMORY_LEAK,
                    intensity=intensity,
                    timestamp=utc_now(),
                    description="Memory leak skipped because maximum allocation was reached",
                    affected_component="memory",
                    impact="No additional memory allocated",
                )

            leak_size_mb = min(size_mb, available_mb)
            allocation: List[bytearray] = []

            try:
                for _ in range(leak_size_mb):
                    allocation.append(bytearray(CHUNK_SIZE))
                    await asyncio.sleep(0)
            except BaseException:
                allocation.clear()
                gc.collect()
                raise

            self._leaked_data.append(allocation)
            self._total_leaked_mb += leak_size_mb

            return ChaosEvent(
                event_id=generate_event_id("memleak"),
                strategy=ChaosStrategy.MEMORY_LEAK,
                intensity=intensity,
                timestamp=utc_now(),
                description=f"Injected memory leak of {leak_size_mb} MB",
                affected_component="memory",
                impact=(
                    f"Allocated {leak_size_mb} MB of memory "
                    f"(total simulated leak: {self._total_leaked_mb} MB)"
                ),
            )

    def get_memory_usage(self) -> Dict[str, Any]:
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "total_leaked_mb": self._total_leaked_mb,
            "leak_allocations": len(self._leaked_data),
            "maximum_leak_mb": self.max_leak_mb,
            "process_rss_mb": memory_info.rss / CHUNK_SIZE,
            "process_vms_mb": memory_info.vms / CHUNK_SIZE,
        }

    def cleanup(self) -> None:
        self._leaked_data.clear()
        self._total_leaked_mb = 0
        gc.collect()


class CPUSpikeSimulator:
    def __init__(
        self,
        max_concurrent_spikes: int = MAX_CONCURRENT_SPIKES,
    ):
        if max_concurrent_spikes <= 0:
            raise ValueError("max_concurrent_spikes must be positive")

        self._max_concurrent_spikes = max_concurrent_spikes
        self._active_spikes = 0
        self._state_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_spikes)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_spikes,
            thread_name_prefix="chaos-cpu",
        )
        self._shutdown = False

    async def inject_cpu_spike(
        self,
        duration: float,
        intensity: ChaosIntensity,
    ) -> ChaosEvent:
        if duration <= 0:
            raise ValueError("duration must be positive")

        if self._shutdown:
            raise RuntimeError("CPU spike simulator is shut down")

        async with self._semaphore:
            async with self._state_lock:
                self._active_spikes += 1

            try:
                loop = asyncio.get_running_loop()

                result = await loop.run_in_executor(
                    self._executor,
                    self._cpu_intensive_task,
                    duration,
                )

                return ChaosEvent(
                    event_id=generate_event_id("cpu"),
                    strategy=ChaosStrategy.CPU_SPIKE,
                    intensity=intensity,
                    timestamp=utc_now(),
                    description=f"Injected CPU spike for {duration:.2f} seconds",
                    affected_component="cpu",
                    duration=duration,
                    impact=result,
                )
            finally:
                async with self._state_lock:
                    self._active_spikes -= 1

    @staticmethod
    def _cpu_intensive_task(duration: float) -> str:
        started = time.monotonic()
        iterations = 0
        accumulator = 0

        while time.monotonic() - started < duration:
            for value in range(5000):
                accumulator = (
                    accumulator * 33 + value * value
                ) & 0xFFFFFFFF

            iterations += 1

        return (
            f"CPU intensive computation completed for "
            f"{duration:.2f} seconds "
            f"({iterations} iterations, checksum {accumulator})"
        )

    def get_active_spikes(self) -> int:
        return self._active_spikes

    def shutdown(self) -> None:
        if self._shutdown:
            return

        self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)


class ChaosInjector:
    def __init__(
        self,
        config: Optional[ChaosConfig] = None,
        max_events: int = MAX_EVENTS_HISTORY,
        max_concurrent_injections: int = MAX_CONCURRENT_INJECTIONS,
    ):
        if max_concurrent_injections <= 0:
            raise ValueError("max_concurrent_injections must be positive")

        self.config = config or ChaosConfig()
        self.monitor = ChaosMonitor(max_events=max_events)

        self.memory_leak_simulator = MemoryLeakSimulator(
            self.config.max_memory_leak_mb
        )
        self.cpu_spike_simulator = CPUSpikeSimulator()

        self._gateway_outages: Dict[str, float] = {}
        self._network_partitions: Dict[str, float] = {}

        self._active_chaos = False
        self._closed = False

        self._injection_semaphore = asyncio.Semaphore(
            max_concurrent_injections
        )

    @property
    def active(self) -> bool:
        return self._active_chaos

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Chaos injector is shut down")

    async def inject_payment_chaos(
        self,
        payment_id: str,
    ) -> List[ChaosEvent]:
        self._ensure_open()

        if not payment_id:
            raise ValueError("payment_id cannot be empty")

        if not self._active_chaos:
            return []

        async with self._injection_semaphore:
            if not self._active_chaos:
                return []

            injected_events: List[ChaosEvent] = []

            for strategy in self.config.enabled_strategies:
                try:
                    event = await self._apply_strategy(
                        strategy,
                        payment_id,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Failed to apply strategy %s",
                        strategy.value,
                    )
                    continue

                if event is None:
                    continue

                injected_events.append(event)
                self.monitor.record_event(event)

            return injected_events

    async def _apply_strategy(
        self,
        strategy: ChaosStrategy,
        payment_id: str,
    ) -> Optional[ChaosEvent]:
        if strategy == ChaosStrategy.RANDOM:
            candidates = [
                candidate
                for candidate in ChaosStrategy
                if candidate != ChaosStrategy.RANDOM
            ]
            selected = random.choice(candidates)

            if random.random() >= self.config.probability_for(selected):
                return None

            return await self._execute_strategy(
                selected,
                payment_id,
            )

        if random.random() >= self.config.probability_for(strategy):
            return None

        return await self._execute_strategy(
            strategy,
            payment_id,
        )

    async def _execute_strategy(
        self,
        strategy: ChaosStrategy,
        payment_id: str,
    ) -> ChaosEvent:
        handlers = {
            ChaosStrategy.LATENCY: self._inject_latency,
            ChaosStrategy.FAILURE: self._inject_failure,
            ChaosStrategy.GATEWAY_OUTAGE: self._inject_gateway_outage,
            ChaosStrategy.MEMORY_LEAK: self._inject_memory_leak,
            ChaosStrategy.CPU_SPIKE: self._inject_cpu_spike,
            ChaosStrategy.NETWORK_PARTITION: self._inject_network_partition,
        }

        handler = handlers.get(strategy)

        if handler is None:
            raise ValueError(f"Unsupported strategy: {strategy.value}")

        return await handler(payment_id)

    async def _inject_latency(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        latency = random.uniform(
            min(0.1, self.config.max_latency),
            self.config.max_latency,
        )

        await asyncio.sleep(latency)

        return ChaosEvent(
            event_id=generate_event_id(ChaosStrategy.LATENCY.value),
            strategy=ChaosStrategy.LATENCY,
            intensity=self.config.intensity,
            timestamp=utc_now(),
            description=f"Injected {latency:.3f} seconds of latency",
            affected_component="network",
            duration=latency,
            impact=f"Payment {payment_id} delayed by {latency:.3f} seconds",
        )

    async def _inject_failure(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        failure_type = random.choice(
            (
                "timeout",
                "connection_reset",
                "protocol_error",
                "server_error",
            )
        )

        return ChaosEvent(
            event_id=generate_event_id(ChaosStrategy.FAILURE.value),
            strategy=ChaosStrategy.FAILURE,
            intensity=self.config.intensity,
            timestamp=utc_now(),
            description=f"Injected {failure_type} failure",
            affected_component="payment_processing",
            impact=f"Payment {payment_id} affected by {failure_type}",
        )

    async def _inject_gateway_outage(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        gateway = random.choice(
            (
                "Stripe",
                "PayPal",
                "Square",
                "Adyen",
            )
        )

        duration = (
            self.config.gateway_outage_duration
            * random.uniform(0.8, 1.2)
        )

        self._gateway_outages[gateway] = (
            time.monotonic() + duration
        )

        return ChaosEvent(
            event_id=generate_event_id(
                ChaosStrategy.GATEWAY_OUTAGE.value
            ),
            strategy=ChaosStrategy.GATEWAY_OUTAGE,
            intensity=self.config.intensity,
            timestamp=utc_now(),
            description=f"Simulated outage for {gateway} gateway",
            affected_component=f"gateway_{gateway.lower()}",
            duration=duration,
            impact=(
                f"Gateway {gateway} marked unavailable while "
                f"processing payment {payment_id}"
            ),
        )

    async def _inject_memory_leak(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        upper_bound = min(
            10,
            self.config.max_memory_leak_mb,
        )

        size_mb = random.randint(1, upper_bound)

        return await self.memory_leak_simulator.inject_memory_leak(
            size_mb,
            self.config.intensity,
        )

    async def _inject_cpu_spike(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        minimum_duration = min(
            0.5,
            self.config.max_cpu_spike_duration,
        )

        duration = random.uniform(
            minimum_duration,
            self.config.max_cpu_spike_duration,
        )

        return await self.cpu_spike_simulator.inject_cpu_spike(
            duration,
            self.config.intensity,
        )

    async def _inject_network_partition(
        self,
        payment_id: str,
    ) -> ChaosEvent:
        component = random.choice(
            (
                "database",
                "cache",
                "external_api",
                "authentication_service",
            )
        )

        duration = (
            self.config.network_partition_duration
            * random.uniform(0.8, 1.2)
        )

        self._network_partitions[component] = (
            time.monotonic() + duration
        )

        return ChaosEvent(
            event_id=generate_event_id(
                ChaosStrategy.NETWORK_PARTITION.value
            ),
            strategy=ChaosStrategy.NETWORK_PARTITION,
            intensity=self.config.intensity,
            timestamp=utc_now(),
            description=(
                f"Simulated network partition for {component}"
            ),
            affected_component=component,
            duration=duration,
            impact=(
                f"Component {component} isolated while "
                f"processing payment {payment_id}"
            ),
        )

    def cleanup_expired_chaos(self) -> None:
        now = time.monotonic()

        self._gateway_outages = {
            gateway: expiry
            for gateway, expiry in self._gateway_outages.items()
            if expiry > now
        }

        self._network_partitions = {
            component: expiry
            for component, expiry in self._network_partitions.items()
            if expiry > now
        }

    def get_success_rate_modifier(self) -> float:
        self.cleanup_expired_chaos()

        modifier = 1.0
        modifier -= 0.1 * len(self._gateway_outages)
        modifier -= 0.05 * len(self._network_partitions)

        return max(modifier, 0.3)

    def is_gateway_available(
        self,
        gateway_name: str,
    ) -> bool:
        self.cleanup_expired_chaos()

        expiry = self._gateway_outages.get(gateway_name)
        return expiry is None

    def enable_chaos(self) -> None:
        self._ensure_open()
        self._active_chaos = True
        logger.info("Chaos injection enabled")

    def disable_chaos(self) -> None:
        self._active_chaos = False
        logger.info("Chaos injection disabled")

    def set_intensity(
        self,
        intensity: ChaosIntensity,
    ) -> None:
        self._ensure_open()

        if not isinstance(intensity, ChaosIntensity):
            raise TypeError("intensity must be a ChaosIntensity")

        self.config = self.config.with_intensity(intensity)

        logger.info(
            "Chaos intensity set to %s",
            intensity.value,
        )

    def get_chaos_metrics(self) -> Dict[str, Any]:
        self.cleanup_expired_chaos()

        base_metrics = self.monitor.get_metrics()

        current_chaos = {
            "active_chaos": self._active_chaos,
            "closed": self._closed,
            "intensity": self.config.intensity.value,
            "active_gateway_outages": len(
                self._gateway_outages
            ),
            "active_network_partitions": len(
                self._network_partitions
            ),
            "success_rate_modifier": (
                self.get_success_rate_modifier()
            ),
            "enabled_strategies": [
                strategy.value
                for strategy in self.config.enabled_strategies
            ],
            "effective_probabilities": {
                strategy.value: self.config.probability_for(strategy)
                for strategy in self.config.enabled_strategies
            },
            "memory_usage": (
                self.memory_leak_simulator.get_memory_usage()
            ),
            "active_cpu_spikes": (
                self.cpu_spike_simulator.get_active_spikes()
            ),
            "gateway_outages": sorted(
                self._gateway_outages.keys()
            ),
            "network_partitions": sorted(
                self._network_partitions.keys()
            ),
        }

        return {
            **base_metrics,
            **current_chaos,
        }

    def shutdown(self) -> None:
        if self._closed:
            return

        self.disable_chaos()
        self.memory_leak_simulator.cleanup()
        self.cpu_spike_simulator.shutdown()
        self._gateway_outages.clear()
        self._network_partitions.clear()
        self._closed = True

        logger.info("Chaos injector shut down")


async def _run_experiment_loop(
    injector: ChaosInjector,
    duration: float,
    events: List[ChaosEvent],
    min_interval: float,
    max_interval: float,
) -> None:
    started = time.monotonic()
    sequence = 0

    while injector.active:
        elapsed = time.monotonic() - started
        remaining = duration - elapsed

        if remaining <= 0:
            break

        interval = min(
            random.uniform(min_interval, max_interval),
            remaining,
        )

        await asyncio.sleep(interval)

        if not injector.active:
            break

        sequence += 1

        injected = await injector.inject_payment_chaos(
            f"experiment_{sequence}"
        )

        events.extend(injected)
        injector.cleanup_expired_chaos()


@asynccontextmanager
async def chaos_experiment_context(
    injector: ChaosInjector,
    duration: float = DEFAULT_EXPERIMENT_DURATION,
    auto_cleanup: bool = True,
    min_interval: float = DEFAULT_MIN_EXPERIMENT_INTERVAL,
    max_interval: float = DEFAULT_MAX_EXPERIMENT_INTERVAL,
) -> AsyncGenerator[List[ChaosEvent], None]:
    if duration <= 0:
        raise ValueError("duration must be positive")

    if min_interval <= 0:
        raise ValueError("min_interval must be positive")

    if max_interval < min_interval:
        raise ValueError(
            "max_interval must be greater than or equal to min_interval"
        )

    injector.enable_chaos()
    experiment_events: List[ChaosEvent] = []

    task = asyncio.create_task(
        _run_experiment_loop(
            injector,
            duration,
            experiment_events,
            min_interval,
            max_interval,
        )
    )

    logger.info(
        "Starting chaos experiment for %.2f seconds",
        duration,
    )

    try:
        yield experiment_events
    finally:
        injector.disable_chaos()

        if not task.done():
            task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        if auto_cleanup:
            injector.memory_leak_simulator.cleanup()

        injector.cleanup_expired_chaos()

        logger.info(
            "Chaos experiment completed with %d events",
            len(experiment_events),
        )


async def run_chaos_experiment(
    injector: ChaosInjector,
    duration: float = DEFAULT_EXPERIMENT_DURATION,
    min_interval: float = DEFAULT_MIN_EXPERIMENT_INTERVAL,
    max_interval: float = DEFAULT_MAX_EXPERIMENT_INTERVAL,
) -> List[ChaosEvent]:
    async with chaos_experiment_context(
        injector,
        duration=duration,
        min_interval=min_interval,
        max_interval=max_interval,
    ) as events:
        await asyncio.sleep(duration)

    return list(events)


def create_chaos_injector(
    intensity: ChaosIntensity = ChaosIntensity.MEDIUM,
    enabled_strategies: Optional[
        Sequence[ChaosStrategy]
    ] = None,
    max_memory_leak_mb: int = 100,
    max_latency: float = 2.0,
    max_cpu_spike_duration: float = 5.0,
) -> ChaosInjector:
    strategies = tuple(
        enabled_strategies
        if enabled_strategies is not None
        else (
            ChaosStrategy.LATENCY,
            ChaosStrategy.FAILURE,
            ChaosStrategy.GATEWAY_OUTAGE,
        )
    )

    config = ChaosConfig(
        intensity=intensity,
        enabled_strategies=strategies,
        max_memory_leak_mb=max_memory_leak_mb,
        max_latency=max_latency,
        max_cpu_spike_duration=max_cpu_spike_duration,
    )

    return ChaosInjector(config)


def parse_strategies(
    values: Iterable[str],
) -> tuple[ChaosStrategy, ...]:
    strategies: List[ChaosStrategy] = []

    for value in values:
        try:
            strategy = ChaosStrategy(value)
        except ValueError as exc:
            valid = ", ".join(
                strategy.value
                for strategy in ChaosStrategy
            )
            raise argparse.ArgumentTypeError(
                f"Unknown strategy '{value}'. Valid values: {valid}"
            ) from exc

        if strategy not in strategies:
            strategies.append(strategy)

    if not strategies:
        raise argparse.ArgumentTypeError(
            "At least one strategy is required"
        )

    return tuple(strategies)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Asynchronous payment chaos experiment simulator"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_EXPERIMENT_DURATION,
    )

    parser.add_argument(
        "--intensity",
        choices=[
            intensity.value
            for intensity in ChaosIntensity
        ],
        default=ChaosIntensity.MEDIUM.value,
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            ChaosStrategy.LATENCY.value,
            ChaosStrategy.FAILURE.value,
            ChaosStrategy.GATEWAY_OUTAGE.value,
        ],
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--request-interval",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-latency",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-cpu-duration",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.duration <= 0:
        parser.error("--duration must be positive")

    if args.requests < 0:
        parser.error("--requests cannot be negative")

    if args.request_interval < 0:
        parser.error("--request-interval cannot be negative")

    if args.max_memory_mb <= 0:
        parser.error("--max-memory-mb must be positive")

    if args.max_latency <= 0:
        parser.error("--max-latency must be positive")

    if args.max_cpu_duration <= 0:
        parser.error("--max-cpu-duration must be positive")


async def run_cli(args: argparse.Namespace) -> int:
    if args.seed is not None:
        random.seed(args.seed)

    strategies = parse_strategies(args.strategies)
    intensity = ChaosIntensity(args.intensity)

    injector = create_chaos_injector(
        intensity=intensity,
        enabled_strategies=strategies,
        max_memory_leak_mb=args.max_memory_mb,
        max_latency=args.max_latency,
        max_cpu_spike_duration=args.max_cpu_duration,
    )

    collected_events: List[ChaosEvent] = []

    try:
        injector.enable_chaos()

        started = time.monotonic()

        for index in range(args.requests):
            if time.monotonic() - started >= args.duration:
                break

            events = await injector.inject_payment_chaos(
                f"payment_{index + 1}"
            )

            collected_events.extend(events)

            if not args.json:
                for event in events:
                    print(
                        f"{event.strategy.value}: "
                        f"{event.description}"
                    )

            if args.request_interval > 0:
                remaining = (
                    args.duration
                    - (time.monotonic() - started)
                )

                if remaining <= 0:
                    break

                await asyncio.sleep(
                    min(args.request_interval, remaining)
                )

        injector.cleanup_expired_chaos()

        metrics = injector.get_chaos_metrics()

        result = {
            "events": [
                event.to_dict()
                for event in collected_events
            ],
            "metrics": metrics,
        }

        if args.json:
            print(
                json.dumps(
                    result,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print()
            print("Chaos experiment summary")
            print("=" * 50)
            print(
                json.dumps(
                    metrics,
                    indent=2,
                    sort_keys=True,
                )
            )

        return 0
    finally:
        injector.shutdown()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.WARNING
        ),
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    try:
        return asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        return 130
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
