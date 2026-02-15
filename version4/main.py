#!/usr/bin/env python3

import asyncio
import random
import time
import tracemalloc
import psutil
import gc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging
import json
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
MAX_EVENTS_HISTORY = 1000
CLEANUP_THRESHOLD_MULTIPLIER = 2.0
MAX_CONCURRENT_SPIKES = 3
EVENT_ID_RANDOM_RANGE = 10000

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

@dataclass
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
    
    enabled_strategies: List[ChaosStrategy] = field(default_factory=lambda: [
        ChaosStrategy.LATENCY,
        ChaosStrategy.FAILURE, 
        ChaosStrategy.GATEWAY_OUTAGE
    ])
    intensity: ChaosIntensity = ChaosIntensity.MEDIUM
    
    gateway_outage_duration: float = 30.0
    network_partition_duration: float = 60.0
    
    def __post_init__(self):
        self._validate()
        self._apply_intensity_multiplier()
    
    def _validate(self):
        if not 0 <= self.latency_probability <= 1:
            raise ValueError("latency_probability must be between 0 and 1")
        if not 0 <= self.failure_probability <= 1:
            raise ValueError("failure_probability must be between 0 and 1")
        if not 0 <= self.gateway_outage_probability <= 1:
            raise ValueError("gateway_outage_probability must be between 0 and 1")
        if not 0 <= self.memory_leak_probability <= 1:
            raise ValueError("memory_leak_probability must be between 0 and 1")
        if not 0 <= self.cpu_spike_probability <= 1:
            raise ValueError("cpu_spike_probability must be between 0 and 1")
        if not 0 <= self.network_partition_probability <= 1:
            raise ValueError("network_partition_probability must be between 0 and 1")
        
        if self.max_latency <= 0:
            raise ValueError("max_latency must be positive")
        if self.max_memory_leak_mb <= 0:
            raise ValueError("max_memory_leak_mb must be positive")
        if self.max_cpu_spike_duration <= 0:
            raise ValueError("max_cpu_spike_duration must be positive")
        if self.gateway_outage_duration <= 0:
            raise ValueError("gateway_outage_duration must be positive")
        if self.network_partition_duration <= 0:
            raise ValueError("network_partition_duration must be positive")
        
        for strategy in self.enabled_strategies:
            if not isinstance(strategy, ChaosStrategy):
                raise TypeError(f"Invalid strategy type: {strategy}")
    
    def _apply_intensity_multiplier(self):
        multiplier = {
            ChaosIntensity.LOW: 0.5,
            ChaosIntensity.MEDIUM: 1.0,
            ChaosIntensity.HIGH: 2.0,
            ChaosIntensity.EXTREME: 4.0
        }.get(self.intensity, 1.0)
        
        self.latency_probability = min(self.latency_probability * multiplier, 1.0)
        self.failure_probability = min(self.failure_probability * multiplier, 1.0)
        self.gateway_outage_probability = min(self.gateway_outage_probability * multiplier, 1.0)
        self.memory_leak_probability = min(self.memory_leak_probability * multiplier, 1.0)
        self.cpu_spike_probability = min(self.cpu_spike_probability * multiplier, 1.0)
        self.network_partition_probability = min(self.network_partition_probability * multiplier, 1.0)

@dataclass
class ChaosEvent:
    event_id: str
    strategy: ChaosStrategy
    intensity: ChaosIntensity
    timestamp: datetime
    description: str
    affected_component: str
    duration: float = 0.0
    impact: str = ""

    def validate(self) -> None:
        if not self.event_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Invalid event_id format")
        if len(self.description) > 500:
            raise ValueError("Description too long")
        if len(self.impact) > 500:
            raise ValueError("Impact too long")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "event_id": self.event_id,
            "strategy": self.strategy.value,
            "intensity": self.intensity.value,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "affected_component": self.affected_component,
            "duration": self.duration,
            "impact": self.impact
        }

class ChaosMonitor:
    def __init__(self, max_events: int = MAX_EVENTS_HISTORY):
        self.events: List[ChaosEvent] = []
        self._event_count = 0
        self._start_time = time.time()
        self._strategy_stats: Dict[ChaosStrategy, Dict[str, Any]] = {}
        self.max_events = max_events
    
    def record_event(self, event: ChaosEvent) -> None:
        event.validate()
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        self._event_count += 1
        
        strategy = event.strategy
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {
                'count': 0,
                'total_duration': 0.0,
                'last_occurrence': event.timestamp
            }
        
        stats = self._strategy_stats[strategy]
        stats['count'] += 1
        stats['total_duration'] += event.duration
        stats['last_occurrence'] = event.timestamp
        
        logger.info(f"Chaos event recorded: {event.description}")
    
    def get_metrics(self) -> Dict[str, Any]:
        current_time = time.time()
        runtime = current_time - self._start_time
        
        events_by_strategy = {}
        for strategy in ChaosStrategy:
            events_by_strategy[strategy.value] = 0
        
        for event in self.events:
            events_by_strategy[event.strategy.value] += 1
        
        strategy_details = {}
        for strategy, stats in self._strategy_stats.items():
            strategy_details[strategy.value] = {
                'total_events': stats['count'],
                'average_duration': stats['total_duration'] / stats['count'] if stats['count'] > 0 else 0,
                'last_occurrence': stats['last_occurrence'].isoformat()
            }
        
        return {
            "total_events": self._event_count,
            "runtime_seconds": runtime,
            "events_per_minute": self._event_count / (runtime / 60) if runtime > 0 else 0,
            "events_by_strategy": events_by_strategy,
            "strategy_details": strategy_details,
            "recent_events": [e.to_dict() for e in self.events[-10:]]
        }
    
    def clear_events(self) -> None:
        self.events.clear()
        self._event_count = 0
        self._strategy_stats.clear()
        self._start_time = time.time()

class MemoryLeakSimulator:
    def __init__(self, max_leak_mb: int = 100):
        self.max_leak_mb = max_leak_mb
        self._leaked_data: List[List[bytearray]] = []
        self._total_leaked = 0
        self._lock = asyncio.Lock()
        
    async def inject_memory_leak(self, size_mb: int) -> ChaosEvent:
        async with self._lock:
            if self._total_leaked >= self.max_leak_mb * CLEANUP_THRESHOLD_MULTIPLIER:
                self.cleanup()
            
            if self._total_leaked >= self.max_leak_mb:
                return ChaosEvent(
                    event_id=self._generate_event_id("memleak_skipped"),
                    strategy=ChaosStrategy.MEMORY_LEAK,
                    intensity=ChaosIntensity.LOW,
                    timestamp=datetime.now(),
                    description="Memory leak skipped - maximum reached",
                    affected_component="memory",
                    impact="No additional memory allocated"
                )
            
            leak_size = min(size_mb, self.max_leak_mb - self._total_leaked)
            
            data = []
            for i in range(leak_size):
                chunk = bytearray(b'X' * CHUNK_SIZE)
                data.append(chunk)
                await asyncio.sleep(0.001)
            
            self._leaked_data.append(data)
            self._total_leaked += leak_size
            
            return ChaosEvent(
                event_id=self._generate_event_id("memleak"),
                strategy=ChaosStrategy.MEMORY_LEAK,
                intensity=ChaosIntensity.MEDIUM,
                timestamp=datetime.now(),
                description=f"Injected memory leak of {leak_size}MB",
                affected_component="memory",
                impact=f"Allocated {leak_size}MB of memory (total: {self._total_leaked}MB)"
            )
    
    def _generate_event_id(self, prefix: str) -> str:
        timestamp = int(time.time() * 1000000)
        random_part = random.randint(1000, 9999)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def get_memory_usage(self) -> Dict[str, Any]:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "total_leaked_mb": self._total_leaked,
            "leak_chunks": len(self._leaked_data),
            "process_rss_mb": memory_info.rss / 1024 / 1024,
            "process_vms_mb": memory_info.vms / 1024 / 1024
        }
    
    def cleanup(self) -> None:
        self._leaked_data.clear()
        self._total_leaked = 0
        gc.collect()

class CPUSpikeSimulator:
    def __init__(self):
        self._active_spikes = 0
        self._max_concurrent_spikes = MAX_CONCURRENT_SPIKES
        self._spike_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SPIKES)
        
    async def inject_cpu_spike(self, duration: float) -> ChaosEvent:
        async with self._spike_lock:
            if self._active_spikes >= self._max_concurrent_spikes:
                return ChaosEvent(
                    event_id=self._generate_event_id("cpu_skip"),
                    strategy=ChaosStrategy.CPU_SPIKE,
                    intensity=ChaosIntensity.LOW,
                    timestamp=datetime.now(),
                    description="CPU spike skipped - too many active spikes",
                    affected_component="cpu",
                    impact="No CPU spike injected due to resource limits"
                )
            self._active_spikes += 1
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, 
                self._cpu_intensive_task, 
                duration
            )
            
            return ChaosEvent(
                event_id=self._generate_event_id("cpu"),
                strategy=ChaosStrategy.CPU_SPIKE,
                intensity=ChaosIntensity.MEDIUM,
                timestamp=datetime.now(),
                description=f"Injected CPU spike for {duration:.2f}s",
                affected_component="cpu",
                duration=duration,
                impact=result
            )
        finally:
            async with self._spike_lock:
                self._active_spikes -= 1
    
    def _generate_event_id(self, prefix: str) -> str:
        timestamp = int(time.time() * 1000000)
        random_part = random.randint(1000, 9999)
        return f"{prefix}_{timestamp}_{random_part}"
    
    def _cpu_intensive_task(self, duration: float) -> str:
        start_time = time.time()
        iterations = 0
        while time.time() - start_time < duration:
            for _ in range(1000):
                _ = [x * x for x in range(1000)]
            iterations += 1
            time.sleep(0.001)
        
        return f"CPU intensive computation for {duration:.2f}s ({iterations} iterations)"
    
    def get_active_spikes(self) -> int:
        return self._active_spikes
    
    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

class ChaosInjector:
    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        self.config.__post_init__()
        self._base_config = ChaosConfig(
            enabled_strategies=self.config.enabled_strategies
        )
        self.monitor = ChaosMonitor()
        self.memory_leak_simulator = MemoryLeakSimulator(self.config.max_memory_leak_mb)
        self.cpu_spike_simulator = CPUSpikeSimulator()
        
        self.gateway_outages: Dict[str, float] = {}
        self.network_partitions: Dict[str, float] = {}
        self._active_chaos = False
        self._injection_lock = asyncio.Lock()
        self._max_concurrent_injections = 5
        self._current_injections = 0
        
        self._strategy_semaphores = {
            ChaosStrategy.CPU_SPIKE: asyncio.Semaphore(1),
            ChaosStrategy.MEMORY_LEAK: asyncio.Semaphore(1),
        }
    
    async def inject_payment_chaos(self, payment_id: str) -> List[ChaosEvent]:
        if not self._active_chaos:
            return []
        
        async with self._injection_lock:
            if self._current_injections >= self._max_concurrent_injections:
                logger.warning(f"Too many concurrent chaos injections for payment {payment_id}")
                return []
            
            self._current_injections += 1
        
        try:
            injected_events = []
            
            for strategy in self.config.enabled_strategies:
                try:
                    event = await self._apply_strategy(strategy, payment_id)
                    if event:
                        injected_events.append(event)
                        self.monitor.record_event(event)
                except Exception as e:
                    logger.error(f"Failed to apply strategy {strategy}: {e}")
            
            return injected_events
        finally:
            async with self._injection_lock:
                self._current_injections -= 1
    
    def _generate_event_id(self, strategy: ChaosStrategy) -> str:
        timestamp = int(time.time() * 1000000)
        random_part = random.randint(1000, 9999)
        return f"{strategy.value}_{timestamp}_{random_part}"
    
    async def _apply_strategy(self, strategy: ChaosStrategy, payment_id: str) -> Optional[ChaosEvent]:
        try:
            if strategy == ChaosStrategy.LATENCY and random.random() < self.config.latency_probability:
                return await self._inject_latency(payment_id)
            
            elif strategy == ChaosStrategy.FAILURE and random.random() < self.config.failure_probability:
                return await self._inject_failure(payment_id)
            
            elif strategy == ChaosStrategy.GATEWAY_OUTAGE and random.random() < self.config.gateway_outage_probability:
                return await self._inject_gateway_outage(payment_id)
            
            elif strategy == ChaosStrategy.MEMORY_LEAK and random.random() < self.config.memory_leak_probability:
                async with self._strategy_semaphores[ChaosStrategy.MEMORY_LEAK]:
                    return await self._inject_memory_leak(payment_id)
            
            elif strategy == ChaosStrategy.CPU_SPIKE and random.random() < self.config.cpu_spike_probability:
                async with self._strategy_semaphores[ChaosStrategy.CPU_SPIKE]:
                    return await self._inject_cpu_spike(payment_id)
            
            elif strategy == ChaosStrategy.NETWORK_PARTITION and random.random() < self.config.network_partition_probability:
                return await self._inject_network_partition(payment_id)
            
            elif strategy == ChaosStrategy.RANDOM:
                available_strategies = [s for s in ChaosStrategy if s != ChaosStrategy.RANDOM]
                random_strategy = random.choice(available_strategies)
                return await self._apply_strategy(random_strategy, payment_id)
        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error applying chaos strategy {strategy}: {e}")
        
        return None
    
    async def _inject_latency(self, payment_id: str) -> ChaosEvent:
        latency = random.uniform(0.1, self.config.max_latency)
        await asyncio.sleep(latency)
        
        return ChaosEvent(
            event_id=self._generate_event_id(ChaosStrategy.LATENCY),
            strategy=ChaosStrategy.LATENCY,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Injected {latency:.3f}s latency",
            affected_component="network",
            duration=latency,
            impact=f"Payment {payment_id} delayed by {latency:.3f}s"
        )
    
    async def _inject_failure(self, payment_id: str) -> ChaosEvent:
        failure_types = ["timeout", "connection_reset", "protocol_error", "server_error"]
        failure_type = random.choice(failure_types)
        
        return ChaosEvent(
            event_id=self._generate_event_id(ChaosStrategy.FAILURE),
            strategy=ChaosStrategy.FAILURE,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Injected {failure_type} failure",
            affected_component="payment_processing",
            duration=0.0,
            impact=f"Payment {payment_id} affected by {failure_type}"
        )
    
    async def _inject_gateway_outage(self, payment_id: str) -> ChaosEvent:
        gateways = ["Stripe", "PayPal", "Square", "Adyen"]
        affected_gateway = random.choice(gateways)
        duration = self.config.gateway_outage_duration * random.uniform(0.8, 1.2)
        outage_end = time.time() + duration
        
        self.gateway_outages[affected_gateway] = outage_end
        
        return ChaosEvent(
            event_id=self._generate_event_id(ChaosStrategy.GATEWAY_OUTAGE),
            strategy=ChaosStrategy.GATEWAY_OUTAGE,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Simulated outage for {affected_gateway} gateway",
            affected_component=f"gateway_{affected_gateway}",
            duration=duration,
            impact=f"Gateway {affected_gateway} marked as unavailable"
        )
    
    async def _inject_memory_leak(self, payment_id: str) -> ChaosEvent:
        leak_size = random.randint(1, 10)
        return await self.memory_leak_simulator.inject_memory_leak(leak_size)
    
    async def _inject_cpu_spike(self, payment_id: str) -> ChaosEvent:
        duration = random.uniform(0.5, self.config.max_cpu_spike_duration)
        return await self.cpu_spike_simulator.inject_cpu_spike(duration)
    
    async def _inject_network_partition(self, payment_id: str) -> ChaosEvent:
        components = ["database", "cache", "external_api", "authentication_service"]
        affected_component = random.choice(components)
        duration = self.config.network_partition_duration * random.uniform(0.8, 1.2)
        partition_end = time.time() + duration
        self.network_partitions[affected_component] = partition_end
        
        return ChaosEvent(
            event_id=self._generate_event_id(ChaosStrategy.NETWORK_PARTITION),
            strategy=ChaosStrategy.NETWORK_PARTITION,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Simulated network partition for {affected_component}",
            affected_component=affected_component,
            duration=duration,
            impact=f"Component {affected_component} isolated from network"
        )
    
    def get_success_rate_modifier(self) -> float:
        modifier = 1.0
        
        active_outages = len([end_time for end_time in self.gateway_outages.values() 
                            if end_time > time.time()])
        modifier -= 0.1 * active_outages
        
        active_partitions = len([end_time for end_time in self.network_partitions.values()
                               if end_time > time.time()])
        modifier -= 0.05 * active_partitions
        
        return max(modifier, 0.3)
    
    def is_gateway_available(self, gateway_name: str) -> bool:
        outage_end = self.gateway_outages.get(gateway_name)
        if outage_end and outage_end > time.time():
            return False
        return True
    
    def cleanup_expired_chaos(self) -> None:
        current_time = time.time()
        
        self.gateway_outages = {
            gateway: end_time 
            for gateway, end_time in self.gateway_outages.items() 
            if end_time > current_time
        }
        
        self.network_partitions = {
            component: end_time
            for component, end_time in self.network_partitions.items()
            if end_time > current_time
        }
    
    def enable_chaos(self) -> None:
        self._active_chaos = True
        logger.info("Chaos injection enabled")
    
    def disable_chaos(self) -> None:
        self._active_chaos = False
        logger.info("Chaos injection disabled")
    
    def set_intensity(self, intensity: ChaosIntensity) -> None:
        new_config = ChaosConfig(
            enabled_strategies=self.config.enabled_strategies,
            intensity=intensity
        )
        new_config.__post_init__()
        self.config = new_config
        logger.info(f"Chaos intensity set to {intensity.value}")
    
    def get_chaos_metrics(self) -> Dict[str, Any]:
        base_metrics = self.monitor.get_metrics()
        
        current_chaos = {
            "active_chaos": self._active_chaos,
            "intensity": self.config.intensity.value,
            "active_gateway_outages": len([end_time for end_time in self.gateway_outages.values() 
                                         if end_time > time.time()]),
            "active_network_partitions": len([end_time for end_time in self.network_partitions.values()
                                           if end_time > time.time()]),
            "success_rate_modifier": self.get_success_rate_modifier(),
            "enabled_strategies": [s.value for s in self.config.enabled_strategies],
            "memory_usage": self.memory_leak_simulator.get_memory_usage(),
            "active_cpu_spikes": self.cpu_spike_simulator.get_active_spikes()
        }
        
        return {**base_metrics, **current_chaos}
    
    def shutdown(self) -> None:
        self.cpu_spike_simulator.shutdown()
        self.memory_leak_simulator.cleanup()

@asynccontextmanager
async def chaos_experiment_context(
    injector: ChaosInjector,
    duration: int = 300,
    auto_cleanup: bool = True
) -> AsyncGenerator[List[ChaosEvent], None]:
    logger.info(f"Starting chaos experiment for {duration} seconds")
    injector.enable_chaos()
    experiment_events: List[ChaosEvent] = []
    
    try:
        start_time = time.time()
        experiment_task = asyncio.create_task(
            _run_experiment_loop(injector, start_time, duration, experiment_events)
        )
        
        yield experiment_events
        
        experiment_task.cancel()
        try:
            await experiment_task
        except asyncio.CancelledError:
            pass
            
    finally:
        injector.disable_chaos()
        if auto_cleanup:
            injector.memory_leak_simulator.cleanup()
        logger.info(f"Chaos experiment completed. Injected {len(experiment_events)} events")

async def _run_experiment_loop(injector: ChaosInjector, start_time: float, duration: int, events_list: List[ChaosEvent]) -> None:
    try:
        while time.time() - start_time < duration:
            await asyncio.sleep(random.uniform(10, 30))
            if injector._active_chaos:
                events = await injector.inject_payment_chaos("experiment")
                events_list.extend(events)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Chaos experiment loop error: {e}")

async def run_chaos_experiment(injector: ChaosInjector, duration: int = 300) -> List[ChaosEvent]:
    events: List[ChaosEvent] = []
    async with chaos_experiment_context(injector, duration) as exp_events:
        events = exp_events
        await asyncio.sleep(duration)
    return events

def create_chaos_injector(
    intensity: ChaosIntensity = ChaosIntensity.MEDIUM,
    enabled_strategies: Optional[List[ChaosStrategy]] = None
) -> ChaosInjector:
    if enabled_strategies is None:
        enabled_strategies = [
            ChaosStrategy.LATENCY,
            ChaosStrategy.FAILURE,
            ChaosStrategy.GATEWAY_OUTAGE
        ]
    
    config = ChaosConfig(
        intensity=intensity,
        enabled_strategies=enabled_strategies
    )
    config.__post_init__()
    
    return ChaosInjector(config)

async def demo_chaos_injector() -> None:
    injector = create_chaos_injector(ChaosIntensity.MEDIUM)
    
    print("Chaos Injector Demo")
    print("=" * 50)
    
    async with chaos_experiment_context(injector, duration=10) as experiment_events:
        for i in range(5):
            events = await injector.inject_payment_chaos(f"demo_payment_{i}")
            for event in events:
                print(f"Chaos Event: {event.description}")
            
            await asyncio.sleep(1)
    
    metrics = injector.get_chaos_metrics()
    memory_usage = injector.memory_leak_simulator.get_memory_usage()
    
    print(f"\nChaos Metrics: {json.dumps(metrics, indent=2)}")
    print(f"\nMemory Usage: {json.dumps(memory_usage, indent=2)}")
    
    injector.shutdown()

if __name__ == "__main__":
    asyncio.run(demo_chaos_injector())
