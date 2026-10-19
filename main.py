#!/usr/bin/env python3

import argparse
import csv
import hashlib
import hmac
import os
import pathlib
import statistics
import tempfile
from dataclasses import asdict
from typing import Any
import asyncio
import gc
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

try:
    import psutil
except ImportError:
    psutil = None

LOG = logging.getLogger("chaosinjector")
MIB = 1024 * 1024


class Strategy(str, Enum):
    LATENCY = "latency"
    FAILURE = "failure"
    GATEWAY_OUTAGE = "gateway_outage"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    NETWORK_PARTITION = "network_partition"
    RANDOM = "random"


class Intensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


MULTIPLIERS = {Intensity.LOW: 0.5, Intensity.MEDIUM: 1.0, Intensity.HIGH: 2.0, Intensity.EXTREME: 4.0}
DEFAULT_PROBABILITIES = {
    Strategy.LATENCY: 0.03,
    Strategy.FAILURE: 0.02,
    Strategy.GATEWAY_OUTAGE: 0.02,
    Strategy.MEMORY_LEAK: 0.01,
    Strategy.CPU_SPIKE: 0.01,
    Strategy.NETWORK_PARTITION: 0.005,
}
GATEWAYS = ("Stripe", "PayPal", "Square", "Adyen")
COMPONENTS = ("database", "cache", "external_api", "authentication_service")
FAILURES = ("timeout", "connection_reset", "protocol_error", "server_error")


@dataclass(frozen=True)
class Config:
    duration: float = 10.0
    requests: int = 5
    request_interval: float = 1.0
    intensity: Intensity = Intensity.MEDIUM
    strategies: tuple[Strategy, ...] = (Strategy.LATENCY, Strategy.FAILURE, Strategy.GATEWAY_OUTAGE)
    max_latency: float = 2.0
    max_memory_mb: int = 100
    max_cpu_duration: float = 5.0
    gateway_outage_duration: float = 30.0
    network_partition_duration: float = 60.0
    max_concurrency: int = 5
    max_cpu_workers: int = 2
    history_limit: int = 1000
    memory_headroom_mb: int = 128
    seed: int | None = None
    dry_run: bool = False
    probabilities: dict[Strategy, float] = field(default_factory=lambda: dict(DEFAULT_PROBABILITIES))

    def __post_init__(self):
        for name in ("duration", "max_latency", "max_cpu_duration", "gateway_outage_duration", "network_partition_duration"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.request_interval) or self.request_interval < 0:
            raise ValueError("request_interval must be finite and nonnegative")
        for name in ("max_memory_mb", "max_concurrency", "max_cpu_workers", "history_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.memory_headroom_mb < 0 or self.requests < 0:
            raise ValueError("memory_headroom_mb and requests must be nonnegative")
        if not self.strategies:
            raise ValueError("at least one strategy is required")
        for strategy, probability in self.probabilities.items():
            if strategy == Strategy.RANDOM or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"invalid probability for {strategy.value}")


@dataclass(frozen=True)
class Event:
    event_id: str
    request_id: str
    strategy: str
    status: str
    timestamp: str
    description: str
    duration: float = 0.0
    component: str = ""
    impact: str = ""

    def as_dict(self):
        return vars(self).copy()


class Monitor:
    def __init__(self, history_limit):
        self.events = deque(maxlen=history_limit)
        self.attempts = Counter()
        self.statuses = Counter()
        self.strategies = Counter()
        self.started = time.monotonic()

    def attempt(self, strategy):
        self.attempts[strategy.value] += 1

    def record(self, event):
        self.events.append(event)
        self.statuses[event.status] += 1
        self.strategies[event.strategy] += 1

    def metrics(self):
        elapsed = time.monotonic() - self.started
        return {
            "runtime_seconds": round(elapsed, 4),
            "attempts_by_strategy": dict(self.attempts),
            "events_by_strategy": dict(self.strategies),
            "events_by_status": dict(self.statuses),
            "total_events": sum(self.statuses.values()),
            "retained_events": len(self.events),
            "events_per_minute": round(sum(self.statuses.values()) * 60 / max(elapsed, 1e-9), 4),
        }


class ChaosInjector:
    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.monitor = Monitor(config.history_limit)
        self.gateway_outages = {}
        self.partitions = {}
        self.memory_chunks = []
        self.allocated_mb = 0
        self.active = False
        self.closed = False
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.cpu_semaphore = asyncio.Semaphore(config.max_cpu_workers)
        self.executor = ThreadPoolExecutor(max_workers=config.max_cpu_workers, thread_name_prefix="chaos-cpu")
        self.stop_cpu = threading.Event()
        self.active_cpu = 0
        self.sequence = 0
        self.deadline = None

    def start(self):
        if self.closed:
            raise RuntimeError("injector is closed")
        self.active = True
        self.deadline = time.monotonic() + self.config.duration

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic()) if self.deadline is not None else 0.0

    def event(self, request_id, strategy, status, description, duration=0.0, component="", impact=""):
        self.sequence += 1
        identifier = f"event_{self.sequence:06d}" if self.config.seed is not None else uuid.uuid4().hex
        result = Event(identifier, request_id, strategy.value, status, datetime.now(timezone.utc).isoformat(), description, round(duration, 4), component, impact)
        self.monitor.record(result)
        return result

    def cleanup_expired(self):
        now = time.monotonic()
        self.gateway_outages = {key: expiry for key, expiry in self.gateway_outages.items() if expiry > now}
        self.partitions = {key: expiry for key, expiry in self.partitions.items() if expiry > now}

    def gateway_available(self, gateway):
        self.cleanup_expired()
        return gateway not in self.gateway_outages

    def component_available(self, component):
        self.cleanup_expired()
        return component not in self.partitions

    def success_rate_modifier(self):
        self.cleanup_expired()
        return max(0.3, 1 - 0.1 * len(self.gateway_outages) - 0.05 * len(self.partitions))

    def memory_available_mb(self):
        if psutil is None:
            return None
        try:
            available = psutil.virtual_memory().available
            return max(0, int(available // MIB) - self.config.memory_headroom_mb)
        except (OSError, RuntimeError, AttributeError):
            return None

    @staticmethod
    def cpu_work(seconds, stop_event):
        started = time.monotonic()
        iterations = 0
        accumulator = 0
        while not stop_event.is_set() and time.monotonic() - started < seconds:
            for value in range(1000):
                accumulator = (accumulator * 33 + value * value) & 0xFFFFFFFF
            iterations += 1
        return iterations, accumulator, time.monotonic() - started

    async def apply(self, strategy, request_id):
        cfg = self.config
        if strategy == Strategy.LATENCY:
            delay = self.rng.uniform(min(0.1, cfg.max_latency), cfg.max_latency)
            actual = min(delay, self.remaining())
            if not cfg.dry_run:
                await asyncio.sleep(actual)
            return self.event(request_id, strategy, "simulated" if cfg.dry_run else "completed", f"Latency of {delay:.3f} seconds", 0 if cfg.dry_run else actual, "network")
        if strategy == Strategy.FAILURE:
            failure = self.rng.choice(FAILURES)
            return self.event(request_id, strategy, "simulated", f"Payment failure: {failure}", component="payment_processing", impact=failure)
        if strategy == Strategy.GATEWAY_OUTAGE:
            gateway = self.rng.choice(GATEWAYS)
            duration = cfg.gateway_outage_duration * self.rng.uniform(0.8, 1.2)
            if not cfg.dry_run:
                self.gateway_outages[gateway] = time.monotonic() + duration
            return self.event(request_id, strategy, "simulated", f"Gateway outage: {gateway}", duration, gateway, "unavailable" if not cfg.dry_run else "dry_run")
        if strategy == Strategy.NETWORK_PARTITION:
            component = self.rng.choice(COMPONENTS)
            duration = cfg.network_partition_duration * self.rng.uniform(0.8, 1.2)
            if not cfg.dry_run:
                self.partitions[component] = time.monotonic() + duration
            return self.event(request_id, strategy, "simulated", f"Network partition: {component}", duration, component, "isolated" if not cfg.dry_run else "dry_run")
        if strategy == Strategy.MEMORY_LEAK:
            desired = self.rng.randint(1, min(10, cfg.max_memory_mb))
            budget = cfg.max_memory_mb - self.allocated_mb
            available = self.memory_available_mb()
            if available is not None:
                budget = min(budget, available)
            size = min(desired, budget)
            if size <= 0:
                return self.event(request_id, strategy, "skipped", "Memory budget exhausted", component="memory")
            if cfg.dry_run:
                return self.event(request_id, strategy, "simulated", f"Would allocate {size} MiB", component="memory")
            chunks = []
            try:
                for _ in range(size):
                    chunks.append(bytearray(MIB))
                    await asyncio.sleep(0)
            except BaseException:
                chunks.clear()
                raise
            self.memory_chunks.extend(chunks)
            self.allocated_mb += size
            return self.event(request_id, strategy, "completed", f"Allocated {size} MiB", component="memory")
        if strategy == Strategy.CPU_SPIKE:
            requested = self.rng.uniform(min(0.5, cfg.max_cpu_duration), cfg.max_cpu_duration)
            duration = min(requested, self.remaining())
            if duration <= 0:
                return self.event(request_id, strategy, "skipped", "Experiment deadline reached", component="cpu")
            if cfg.dry_run:
                return self.event(request_id, strategy, "simulated", f"Would run CPU load for {duration:.3f} seconds", component="cpu")
            async with self.cpu_semaphore:
                self.active_cpu += 1
                try:
                    loop = asyncio.get_running_loop()
                    iterations, checksum, elapsed = await loop.run_in_executor(self.executor, self.cpu_work, duration, self.stop_cpu)
                finally:
                    self.active_cpu -= 1
            return self.event(request_id, strategy, "completed", f"CPU load ran for {elapsed:.3f} seconds", elapsed, "cpu", f"iterations={iterations}, checksum={checksum}")
        raise ValueError(f"unsupported strategy: {strategy.value}")

    async def inject_payment_chaos(self, request_id):
        if not self.active or self.closed or self.remaining() <= 0:
            return []
        events = []
        async with self.semaphore:
            for chosen in self.config.strategies:
                if not self.active or self.remaining() <= 0:
                    break
                strategy = self.rng.choice(tuple(DEFAULT_PROBABILITIES)) if chosen == Strategy.RANDOM else chosen
                self.monitor.attempt(strategy)
                probability = min(1.0, self.config.probabilities[strategy] * MULTIPLIERS[self.config.intensity])
                if self.rng.random() >= probability:
                    continue
                try:
                    event = await self.apply(strategy, request_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOG.exception("Strategy failed: %s", strategy.value)
                    event = self.event(request_id, strategy, "failed", type(exc).__name__, component=strategy.value)
                events.append(event)
        return events

    def metrics(self):
        self.cleanup_expired()
        data = self.monitor.metrics()
        data.update({
            "intensity": self.config.intensity.value,
            "dry_run": self.config.dry_run,
            "seed": self.config.seed,
            "active": self.active,
            "active_cpu_workers": self.active_cpu,
            "allocated_memory_mb": self.allocated_mb,
            "gateway_outages": sorted(self.gateway_outages),
            "network_partitions": sorted(self.partitions),
            "success_rate_modifier": self.success_rate_modifier(),
            "effective_probabilities": {s.value: min(1.0, p * MULTIPLIERS[self.config.intensity]) for s, p in self.config.probabilities.items()},
        })
        if psutil is not None:
            try:
                data["process_rss_mb"] = round(psutil.Process().memory_info().rss / MIB, 2)
            except (OSError, RuntimeError, AttributeError):
                pass
        return data

    def close(self):
        if self.closed:
            return
        self.active = False
        self.stop_cpu.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.memory_chunks.clear()
        self.allocated_mb = 0
        gc.collect()
        self.gateway_outages.clear()
        self.partitions.clear()
        self.closed = True


async def run_experiment(config):
    injector = ChaosInjector(config)
    events = []
    try:
        injector.start()
        for index in range(config.requests):
            if injector.remaining() <= 0:
                break
            events.extend(await injector.inject_payment_chaos(f"payment_{index + 1}"))
            if index + 1 < config.requests and config.request_interval > 0:
                await asyncio.sleep(min(config.request_interval, injector.remaining()))
        metrics = injector.metrics()
        return {"events": [event.as_dict() for event in events], "metrics": metrics}
    finally:
        injector.close()


def parse_probability(value):
    try:
        name, raw = value.split("=", 1)
        strategy = Strategy(name)
        probability = float(raw)
        if strategy == Strategy.RANDOM or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError
        return strategy, probability
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected STRATEGY=PROBABILITY, with probability between 0 and 1") from exc


def build_parser():
    parser = argparse.ArgumentParser(description="Bounded asynchronous payment chaos simulator")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--intensity", choices=[x.value for x in Intensity], default="medium")
    parser.add_argument("--strategies", nargs="+", choices=[x.value for x in Strategy], default=["latency", "failure", "gateway_outage"])
    parser.add_argument("--probability", type=parse_probability, action="append", default=[], metavar="STRATEGY=P")
    parser.add_argument("--max-memory-mb", type=int, default=100)
    parser.add_argument("--memory-headroom-mb", type=int, default=128)
    parser.add_argument("--max-latency", type=float, default=2.0)
    parser.add_argument("--max-cpu-duration", type=float, default=5.0)
    parser.add_argument("--gateway-outage-duration", type=float, default=30.0)
    parser.add_argument("--network-partition-duration", type=float, default=60.0)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scenario", choices=[s.name for s in built_in_scenarios()])
    parser.add_argument("--scenario-file")
    parser.add_argument("--report-format", choices=[x.value for x in ReportFormat], default="text")
    parser.add_argument("--output")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--verify-result")
    parser.add_argument("--archive-directory")
    parser.add_argument("--archive-name")
    return parser


class ExperimentError(Exception):
    pass


class PolicyViolation(ExperimentError):
    pass


class InvalidScenario(ExperimentError):
    pass


class StorageError(ExperimentError):
    pass


class VerificationError(ExperimentError):
    pass


class Lifecycle(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    DEFER = "defer"


class ReportFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TEXT = "text"
    MARKDOWN = "markdown"


class ScheduleMode(str, Enum):
    FIXED = "fixed"
    POISSON = "poisson"
    JITTER = "jitter"
    BURST = "burst"


class RequestStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    message: str
    subject: str = ""
    observed: Any = None
    expected: Any = None

    def as_dict(self):
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "subject": self.subject,
            "observed": self.observed,
            "expected": self.expected,
        }


@dataclass(frozen=True)
class Threshold:
    name: str
    minimum: float | None = None
    maximum: float | None = None
    severity: Severity = Severity.WARNING

    def __post_init__(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("threshold must have a bound")
        for value in (self.minimum, self.maximum):
            if value is not None and not math.isfinite(value):
                raise ValueError("threshold bound must be finite")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum cannot exceed maximum")

    def evaluate(self, value):
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            return Finding(
                "invalid_metric",
                Severity.ERROR,
                "Metric is missing or nonfinite",
                self.name,
                value,
            )
        if self.minimum is not None and value < self.minimum:
            return Finding(
                "below_threshold",
                self.severity,
                "Metric fell below minimum",
                self.name,
                value,
                self.minimum,
            )
        if self.maximum is not None and value > self.maximum:
            return Finding(
                "above_threshold",
                self.severity,
                "Metric exceeded maximum",
                self.name,
                value,
                self.maximum,
            )
        return None


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: str
    monotonic_time: float
    rss_bytes: int
    virtual_bytes: int
    available_bytes: int | None
    cpu_percent: float | None
    threads: int | None
    open_fds: int | None

    def as_dict(self):
        return asdict(self)


class ResourceSampler:
    def __init__(self, max_samples=4096):
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self.samples = deque(maxlen=max_samples)
        self.process = psutil.Process() if psutil is not None else None

    def capture(self):
        rss = 0
        virtual = 0
        available = None
        cpu = None
        threads = None
        fds = None
        if self.process is not None:
            try:
                memory = self.process.memory_info()
                rss = memory.rss
                virtual = memory.vms
                cpu = self.process.cpu_percent(interval=None)
                threads = self.process.num_threads()
                if hasattr(self.process, "num_fds"):
                    fds = self.process.num_fds()
                available = psutil.virtual_memory().available
            except (OSError, RuntimeError, psutil.Error):
                pass
        sample = ResourceSnapshot(
            datetime.now(timezone.utc).isoformat(),
            time.monotonic(),
            rss,
            virtual,
            available,
            cpu,
            threads,
            fds,
        )
        self.samples.append(sample)
        return sample

    def summary(self):
        if not self.samples:
            return {"samples": 0}
        rss = [sample.rss_bytes for sample in self.samples]
        virtual = [sample.virtual_bytes for sample in self.samples]
        cpu = [sample.cpu_percent for sample in self.samples if sample.cpu_percent is not None]
        return {
            "samples": len(self.samples),
            "rss_min_bytes": min(rss),
            "rss_max_bytes": max(rss),
            "rss_mean_bytes": statistics.fmean(rss),
            "virtual_max_bytes": max(virtual),
            "cpu_mean_percent": statistics.fmean(cpu) if cpu else None,
            "duration_seconds": max(
                0.0,
                self.samples[-1].monotonic_time - self.samples[0].monotonic_time,
            ),
        }

    def clear(self):
        self.samples.clear()


class Histogram:
    def __init__(self, boundaries=None):
        values = boundaries or (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
        self.boundaries = tuple(sorted(set(float(value) for value in values)))
        if not self.boundaries or any(value <= 0 for value in self.boundaries):
            raise ValueError("histogram boundaries must be positive")
        self.buckets = [0] * (len(self.boundaries) + 1)
        self.values = []

    def observe(self, value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("histogram observation must be nonnegative and finite")
        self.values.append(value)
        for index, boundary in enumerate(self.boundaries):
            if value <= boundary:
                self.buckets[index] += 1
                return
        self.buckets[-1] += 1

    def percentile(self, percentile):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        if not self.values:
            return None
        ordered = sorted(self.values)
        position = (len(ordered) - 1) * percentile / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    def summary(self):
        return {
            "count": len(self.values),
            "sum": sum(self.values),
            "min": min(self.values) if self.values else None,
            "max": max(self.values) if self.values else None,
            "mean": statistics.fmean(self.values) if self.values else None,
            "p50": self.percentile(50),
            "p90": self.percentile(90),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "boundaries": list(self.boundaries),
            "buckets": list(self.buckets),
        }

    def merge(self, other):
        if self.boundaries != other.boundaries:
            raise ValueError("histogram boundaries do not match")
        for value in other.values:
            self.observe(value)


class CounterRegistry:
    def __init__(self):
        self.counters = Counter()
        self.gauges = {}
        self.histograms = {}

    def increment(self, name, amount=1, labels=None):
        if amount < 0:
            raise ValueError("counter increment must be nonnegative")
        key = self._key(name, labels)
        self.counters[key] += amount

    def set_gauge(self, name, value, labels=None):
        if not math.isfinite(float(value)):
            raise ValueError("gauge must be finite")
        self.gauges[self._key(name, labels)] = float(value)

    def observe(self, name, value, labels=None):
        key = self._key(name, labels)
        self.histograms.setdefault(key, Histogram()).observe(value)

    @staticmethod
    def _key(name, labels):
        if not name or not isinstance(name, str):
            raise ValueError("metric name is required")
        suffix = tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))
        return name, suffix

    @staticmethod
    def _label_key(key):
        name, labels = key
        if not labels:
            return name
        return name + "{" + ",".join(f"{k}={v}" for k, v in labels) + "}"

    def snapshot(self):
        return {
            "counters": {
                self._label_key(key): value
                for key, value in sorted(self.counters.items())
            },
            "gauges": {
                self._label_key(key): value
                for key, value in sorted(self.gauges.items())
            },
            "histograms": {
                self._label_key(key): value.summary()
                for key, value in sorted(self.histograms.items())
            },
        }


class StateMachine:
    ALLOWED = {
        Lifecycle.CREATED: {Lifecycle.VALIDATED, Lifecycle.CANCELLED, Lifecycle.FAILED},
        Lifecycle.VALIDATED: {Lifecycle.READY, Lifecycle.CANCELLED, Lifecycle.FAILED},
        Lifecycle.READY: {Lifecycle.RUNNING, Lifecycle.CANCELLED, Lifecycle.FAILED},
        Lifecycle.RUNNING: {Lifecycle.STOPPING, Lifecycle.FAILED, Lifecycle.CANCELLED},
        Lifecycle.STOPPING: {Lifecycle.COMPLETED, Lifecycle.FAILED, Lifecycle.CANCELLED},
        Lifecycle.COMPLETED: set(),
        Lifecycle.FAILED: set(),
        Lifecycle.CANCELLED: set(),
    }

    def __init__(self):
        self.state = Lifecycle.CREATED
        self.transitions = []

    def transition(self, target, reason=""):
        target = Lifecycle(target)
        if target not in self.ALLOWED[self.state]:
            raise ExperimentError(f"invalid transition {self.state.value} -> {target.value}")
        record = {
            "from": self.state.value,
            "to": target.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.transitions.append(record)
        self.state = target
        return record

    def terminal(self):
        return self.state in (Lifecycle.COMPLETED, Lifecycle.FAILED, Lifecycle.CANCELLED)

    def snapshot(self):
        return {"state": self.state.value, "transitions": list(self.transitions)}


@dataclass(frozen=True)
class SafetyPolicy:
    max_duration: float = 3600.0
    max_requests: int = 100000
    max_memory_mb: int = 1024
    max_cpu_workers: int = 8
    max_cpu_duration: float = 60.0
    max_concurrency: int = 64
    require_dry_run_for_extreme: bool = True
    allow_real_cpu: bool = True
    allow_real_memory: bool = True

    def check(self, config):
        findings = []
        comparisons = (
            ("duration", config.duration, self.max_duration),
            ("requests", config.requests, self.max_requests),
            ("max_memory_mb", config.max_memory_mb, self.max_memory_mb),
            ("max_cpu_workers", config.max_cpu_workers, self.max_cpu_workers),
            ("max_cpu_duration", config.max_cpu_duration, self.max_cpu_duration),
            ("max_concurrency", config.max_concurrency, self.max_concurrency),
        )
        for name, observed, maximum in comparisons:
            if observed > maximum:
                findings.append(Finding(
                    "policy_limit",
                    Severity.ERROR,
                    f"{name} exceeds safety policy",
                    name,
                    observed,
                    maximum,
                ))
        if self.require_dry_run_for_extreme:
            if config.intensity == Intensity.EXTREME and not config.dry_run:
                findings.append(Finding(
                    "extreme_requires_dry_run",
                    Severity.ERROR,
                    "Extreme intensity requires dry-run mode",
                    "intensity",
                ))
        if not config.dry_run:
            if not self.allow_real_cpu and Strategy.CPU_SPIKE in config.strategies:
                findings.append(Finding(
                    "cpu_not_permitted",
                    Severity.ERROR,
                    "CPU workloads are disabled by policy",
                    "cpu_spike",
                ))
            if not self.allow_real_memory and Strategy.MEMORY_LEAK in config.strategies:
                findings.append(Finding(
                    "memory_not_permitted",
                    Severity.ERROR,
                    "Memory allocations are disabled by policy",
                    "memory_leak",
                ))
        return findings

    def enforce(self, config):
        findings = self.check(config)
        if findings:
            raise PolicyViolation("; ".join(f.message for f in findings))
        return True


class RateLimiter:
    def __init__(self, rate, capacity):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.updated = time.monotonic()

    def replenish(self, now=None):
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now

    def acquire(self, count=1, now=None):
        if count <= 0:
            raise ValueError("count must be positive")
        self.replenish(now)
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    def wait_time(self, count=1, now=None):
        self.replenish(now)
        if count > self.capacity:
            return math.inf
        return max(0.0, count - self.tokens) / self.rate

    def snapshot(self):
        self.replenish()
        return {"rate": self.rate, "capacity": self.capacity, "tokens": self.tokens}


class CircuitBreaker:
    def __init__(self, threshold=5, recovery_seconds=10.0):
        if threshold < 1 or recovery_seconds <= 0:
            raise ValueError("invalid circuit breaker settings")
        self.threshold = threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at = None
        self.half_open = False

    def permit(self, now=None):
        now = time.monotonic() if now is None else now
        if self.opened_at is None:
            return True
        if now - self.opened_at >= self.recovery_seconds:
            self.half_open = True
            return True
        return False

    def success(self):
        self.failures = 0
        self.opened_at = None
        self.half_open = False

    def failure(self, now=None):
        now = time.monotonic() if now is None else now
        self.failures += 1
        if self.half_open or self.failures >= self.threshold:
            self.opened_at = now
            self.half_open = False

    def snapshot(self):
        state = "half_open" if self.half_open else "open" if self.opened_at is not None else "closed"
        return {
            "state": state,
            "failures": self.failures,
            "threshold": self.threshold,
            "recovery_seconds": self.recovery_seconds,
        }


class Backoff:
    def __init__(self, initial=0.1, factor=2.0, maximum=10.0, jitter=0.1, seed=None):
        if initial <= 0 or factor < 1 or maximum < initial or not 0 <= jitter <= 1:
            raise ValueError("invalid backoff parameters")
        self.initial = initial
        self.factor = factor
        self.maximum = maximum
        self.jitter = jitter
        self.rng = random.Random(seed)

    def delay(self, attempt):
        if attempt < 0:
            raise ValueError("attempt must be nonnegative")
        raw = min(self.maximum, self.initial * self.factor ** min(attempt, 100))
        return max(0.0, raw * self.rng.uniform(1 - self.jitter, 1 + self.jitter))

    def sequence(self, count):
        if count < 0:
            raise ValueError("count must be nonnegative")
        return [self.delay(index) for index in range(count)]


class FaultRegistry:
    def __init__(self):
        self.outages = {}
        self.partitions = {}
        self.failures = {}

    def register_outage(self, gateway, duration, now=None):
        if not gateway or duration <= 0:
            raise ValueError("invalid outage")
        now = time.monotonic() if now is None else now
        self.outages[gateway] = max(self.outages.get(gateway, 0), now + duration)

    def register_partition(self, component, duration, now=None):
        if not component or duration <= 0:
            raise ValueError("invalid partition")
        now = time.monotonic() if now is None else now
        self.partitions[component] = max(self.partitions.get(component, 0), now + duration)

    def register_failure(self, request_id, failure_type):
        if not request_id:
            raise ValueError("request id required")
        self.failures[request_id] = failure_type

    def cleanup(self, now=None):
        now = time.monotonic() if now is None else now
        self.outages = {key: expiry for key, expiry in self.outages.items() if expiry > now}
        self.partitions = {key: expiry for key, expiry in self.partitions.items() if expiry > now}

    def gateway_available(self, gateway, now=None):
        self.cleanup(now)
        return gateway not in self.outages

    def component_available(self, component, now=None):
        self.cleanup(now)
        return component not in self.partitions

    def request_failure(self, request_id):
        return self.failures.get(request_id)

    def consume_failure(self, request_id):
        return self.failures.pop(request_id, None)

    def snapshot(self, now=None):
        self.cleanup(now)
        return {
            "outages": dict(self.outages),
            "partitions": dict(self.partitions),
            "pending_failures": dict(self.failures),
        }


@dataclass(frozen=True)
class PaymentRequest:
    request_id: str
    gateway: str
    component: str
    amount: float
    currency: str = "EUR"
    timeout_seconds: float = 5.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.request_id or not self.gateway or not self.component:
            raise ValueError("payment request fields cannot be empty")
        if not math.isfinite(self.amount) or self.amount < 0:
            raise ValueError("amount must be finite and nonnegative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PaymentResult:
    request_id: str
    status: RequestStatus
    gateway: str
    elapsed_seconds: float
    reason: str
    amount: float
    currency: str

    def as_dict(self):
        data = asdict(self)
        data["status"] = self.status.value
        return data


class PaymentSimulator:
    def __init__(self, registry=None, seed=None, baseline_latency=0.01):
        if baseline_latency < 0:
            raise ValueError("baseline_latency must be nonnegative")
        self.registry = registry or FaultRegistry()
        self.rng = random.Random(seed)
        self.baseline_latency = baseline_latency
        self.results = []

    async def process(self, request, dry_run=False):
        started = time.monotonic()
        failure = self.registry.consume_failure(request.request_id)
        if not self.registry.gateway_available(request.gateway):
            status = RequestStatus.BLOCKED
            reason = "gateway_outage"
        elif not self.registry.component_available(request.component):
            status = RequestStatus.BLOCKED
            reason = "network_partition"
        elif failure == "timeout":
            status = RequestStatus.TIMEOUT
            reason = failure
        elif failure is not None:
            status = RequestStatus.REJECTED
            reason = failure
        else:
            status = RequestStatus.ACCEPTED
            reason = "processed"
        if not dry_run and self.baseline_latency:
            await asyncio.sleep(min(self.baseline_latency, request.timeout_seconds))
        elapsed = time.monotonic() - started
        result = PaymentResult(
            request.request_id,
            status,
            request.gateway,
            elapsed,
            reason,
            request.amount,
            request.currency,
        )
        self.results.append(result)
        return result

    def metrics(self):
        counts = Counter(result.status.value for result in self.results)
        total = len(self.results)
        durations = [result.elapsed_seconds for result in self.results]
        return {
            "total": total,
            "statuses": dict(counts),
            "acceptance_rate": counts[RequestStatus.ACCEPTED.value] / total if total else None,
            "mean_latency": statistics.fmean(durations) if durations else None,
            "total_amount_accepted": sum(
                result.amount for result in self.results
                if result.status == RequestStatus.ACCEPTED
            ),
        }


@dataclass(frozen=True)
class ScenarioStep:
    name: str
    strategy: Strategy
    probability: float = 1.0
    repeat: int = 1
    pause_seconds: float = 0.0
    target: str = ""

    def __post_init__(self):
        if not self.name:
            raise InvalidScenario("step name required")
        if not isinstance(self.strategy, Strategy):
            raise InvalidScenario("invalid strategy")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise InvalidScenario("probability out of range")
        if self.repeat < 1:
            raise InvalidScenario("repeat must be positive")
        if self.pause_seconds < 0 or not math.isfinite(self.pause_seconds):
            raise InvalidScenario("pause must be nonnegative and finite")

    def as_dict(self):
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "probability": self.probability,
            "repeat": self.repeat,
            "pause_seconds": self.pause_seconds,
            "target": self.target,
        }


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[ScenarioStep, ...]
    description: str = ""
    tags: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self):
        if not self.name:
            raise InvalidScenario("scenario name required")
        if not self.steps:
            raise InvalidScenario("scenario must contain steps")
        if self.version < 1:
            raise InvalidScenario("scenario version must be positive")
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise InvalidScenario("step names must be unique")

    def as_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": list(self.tags),
            "steps": [step.as_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise InvalidScenario("scenario must be an object")
        steps = []
        for item in data.get("steps", []):
            if not isinstance(item, dict):
                raise InvalidScenario("step must be an object")
            try:
                step = ScenarioStep(
                    name=item["name"],
                    strategy=Strategy(item["strategy"]),
                    probability=float(item.get("probability", 1.0)),
                    repeat=int(item.get("repeat", 1)),
                    pause_seconds=float(item.get("pause_seconds", 0.0)),
                    target=str(item.get("target", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidScenario(f"invalid scenario step: {exc}") from exc
            steps.append(step)
        return cls(
            name=data.get("name", ""),
            steps=tuple(steps),
            description=data.get("description", ""),
            tags=tuple(data.get("tags", [])),
            version=int(data.get("version", 1)),
        )


class ScenarioLibrary:
    def __init__(self):
        self.scenarios = {}

    def add(self, scenario, overwrite=False):
        if scenario.name in self.scenarios and not overwrite:
            raise InvalidScenario(f"scenario already exists: {scenario.name}")
        self.scenarios[scenario.name] = scenario

    def get(self, name):
        if name not in self.scenarios:
            raise InvalidScenario(f"unknown scenario: {name}")
        return self.scenarios[name]

    def remove(self, name):
        return self.scenarios.pop(name)

    def names(self):
        return sorted(self.scenarios)

    def load(self, path):
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            scenarios = [Scenario.from_dict(item) for item in data]
        else:
            scenarios = [Scenario.from_dict(data)]
        for scenario in scenarios:
            self.add(scenario)
        return scenarios

    def save(self, path):
        data = [self.scenarios[name].as_dict() for name in self.names()]
        atomic_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


class ScenarioRunner:
    def __init__(self, injector, seed=None):
        self.injector = injector
        self.rng = random.Random(seed)
        self.executed = []

    async def execute(self, scenario):
        if not self.injector.active:
            raise ExperimentError("injector must be started")
        for step in scenario.steps:
            for index in range(step.repeat):
                if self.injector.remaining() <= 0:
                    return list(self.executed)
                if self.rng.random() > step.probability:
                    continue
                strategy = step.strategy
                if strategy == Strategy.RANDOM:
                    strategy = self.rng.choice(tuple(DEFAULT_PROBABILITIES))
                request_id = f"{scenario.name}:{step.name}:{index + 1}"
                self.injector.monitor.attempt(strategy)
                event = await self.injector.apply(strategy, request_id)
                self.executed.append(event)
                if step.pause_seconds:
                    await asyncio.sleep(min(step.pause_seconds, self.injector.remaining()))
        return list(self.executed)


class ScheduleGenerator:
    def __init__(self, mode=ScheduleMode.FIXED, interval=1.0, jitter=0.25, seed=None):
        self.mode = ScheduleMode(mode)
        if interval <= 0 or not math.isfinite(interval):
            raise ValueError("interval must be positive and finite")
        if not 0 <= jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")
        self.interval = interval
        self.jitter = jitter
        self.rng = random.Random(seed)

    def next_interval(self, index=0):
        if self.mode == ScheduleMode.FIXED:
            return self.interval
        if self.mode == ScheduleMode.POISSON:
            return self.rng.expovariate(1.0 / self.interval)
        if self.mode == ScheduleMode.JITTER:
            return self.interval * self.rng.uniform(1 - self.jitter, 1 + self.jitter)
        if self.mode == ScheduleMode.BURST:
            return self.interval * (0.1 if index % 5 < 4 else 4.6)
        raise ValueError("unsupported schedule mode")

    def generate(self, count):
        if count < 0:
            raise ValueError("count must be nonnegative")
        return [self.next_interval(index) for index in range(count)]


class EventFilter:
    def __init__(self, strategies=None, statuses=None, components=None, text=None):
        self.strategies = set(strategies or [])
        self.statuses = set(statuses or [])
        self.components = set(components or [])
        self.text = text.casefold() if text else None

    def matches(self, event):
        data = event.as_dict() if hasattr(event, "as_dict") else event
        if self.strategies and data.get("strategy") not in self.strategies:
            return False
        if self.statuses and data.get("status") not in self.statuses:
            return False
        if self.components and data.get("component") not in self.components:
            return False
        if self.text:
            haystack = " ".join(str(value) for value in data.values()).casefold()
            if self.text not in haystack:
                return False
        return True

    def select(self, events):
        return [event for event in events if self.matches(event)]


class EventAggregator:
    def __init__(self, events=()):
        self.events = list(events)

    def add(self, event):
        self.events.append(event)

    def group_by(self, field_name):
        groups = {}
        for event in self.events:
            data = event.as_dict() if hasattr(event, "as_dict") else event
            key = str(data.get(field_name, ""))
            groups.setdefault(key, []).append(data)
        return groups

    def counts(self, field_name):
        return {key: len(value) for key, value in self.group_by(field_name).items()}

    def duration_statistics(self):
        values = []
        for event in self.events:
            data = event.as_dict() if hasattr(event, "as_dict") else event
            duration = data.get("duration", 0)
            if isinstance(duration, (int, float)) and math.isfinite(duration):
                values.append(duration)
        histogram = Histogram()
        for value in values:
            if value >= 0:
                histogram.observe(value)
        return histogram.summary()

    def summary(self):
        return {
            "total": len(self.events),
            "by_strategy": self.counts("strategy"),
            "by_status": self.counts("status"),
            "by_component": self.counts("component"),
            "duration": self.duration_statistics(),
        }


class InvariantChecker:
    def __init__(self):
        self.thresholds = []
        self.custom = []

    def add_threshold(self, threshold):
        self.thresholds.append(threshold)

    def add_check(self, name, callback):
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.custom.append((name, callback))

    def evaluate(self, metrics):
        findings = []
        for threshold in self.thresholds:
            value = metrics.get(threshold.name)
            finding = threshold.evaluate(value)
            if finding is not None:
                findings.append(finding)
        for name, callback in self.custom:
            try:
                outcome = callback(metrics)
                if isinstance(outcome, Finding):
                    findings.append(outcome)
                elif outcome is False:
                    findings.append(Finding(
                        "custom_check_failed",
                        Severity.ERROR,
                        f"Custom invariant failed: {name}",
                        name,
                    ))
            except Exception as exc:
                findings.append(Finding(
                    "custom_check_error",
                    Severity.ERROR,
                    f"Custom invariant raised {type(exc).__name__}",
                    name,
                ))
        return findings


class AuditTrail:
    def __init__(self, secret=None):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.entries = []
        self.last_digest = "0" * 64

    def append(self, action, payload):
        if not action:
            raise ValueError("audit action required")
        entry = {
            "sequence": len(self.entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "payload": payload,
            "previous_digest": self.last_digest,
        }
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        if self.secret is None:
            digest = hashlib.sha256(encoded).hexdigest()
        else:
            digest = hmac.new(self.secret, encoded, hashlib.sha256).hexdigest()
        entry["digest"] = digest
        self.entries.append(entry)
        self.last_digest = digest
        return entry

    def verify(self):
        previous = "0" * 64
        for index, entry in enumerate(self.entries, 1):
            if entry.get("sequence") != index:
                return False
            if entry.get("previous_digest") != previous:
                return False
            unsigned = {key: value for key, value in entry.items() if key != "digest"}
            encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
            if self.secret is None:
                expected = hashlib.sha256(encoded).hexdigest()
            else:
                expected = hmac.new(self.secret, encoded, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, entry.get("digest", "")):
                return False
            previous = expected
        return True

    def export(self):
        return {"entries": list(self.entries), "valid": self.verify(), "digest": self.last_digest}


class JsonCodec:
    @staticmethod
    def encode(value, pretty=True):
        return json.dumps(
            value,
            indent=2 if pretty else None,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=JsonCodec._default,
        )

    @staticmethod
    def _default(value):
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "as_dict"):
            return value.as_dict()
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unsupported JSON type: {type(value).__name__}")

    @staticmethod
    def decode(value):
        return json.loads(value)


class JsonlCodec:
    @staticmethod
    def encode(events):
        return "".join(JsonCodec.encode(event, pretty=False) + "\n" for event in events)

    @staticmethod
    def decode(content):
        return [json.loads(line) for line in content.splitlines() if line.strip()]


class TextReporter:
    @staticmethod
    def render(result):
        metrics = result.get("metrics", {})
        events = result.get("events", [])
        lines = ["Chaos Injector Experiment", "=" * 48]
        lines.append(f"Events: {len(events)}")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (dict, list)):
                value = JsonCodec.encode(value, pretty=False)
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("Event Timeline")
        lines.append("-" * 48)
        for event in events:
            lines.append(
                f"{event.get('timestamp', '')} "
                f"{event.get('strategy', ''):<20} "
                f"{event.get('status', ''):<10} "
                f"{event.get('description', '')}"
            )
        return "\n".join(lines) + "\n"


class MarkdownReporter:
    @staticmethod
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    @classmethod
    def render(cls, result):
        metrics = result.get("metrics", {})
        events = result.get("events", [])
        lines = ["# Chaos Injector Experiment", "", "## Metrics", "", "| Metric | Value |", "|---|---|"]
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (dict, list)):
                value = JsonCodec.encode(value, pretty=False)
            lines.append(f"| {cls.escape(key)} | {cls.escape(value)} |")
        lines.extend(("", "## Events", "", "| Timestamp | Strategy | Status | Description |", "|---|---|---|---|"))
        for event in events:
            cells = [cls.escape(event.get(key, "")) for key in ("timestamp", "strategy", "status", "description")]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


class CsvReporter:
    FIELDS = (
        "event_id",
        "request_id",
        "strategy",
        "status",
        "timestamp",
        "description",
        "duration",
        "component",
        "impact",
    )

    @classmethod
    def render(cls, result):
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=cls.FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.get("events", []))
        return output.getvalue()


class ReportFactory:
    @staticmethod
    def render(result, report_format):
        report_format = ReportFormat(report_format)
        if report_format == ReportFormat.JSON:
            return JsonCodec.encode(result) + "\n"
        if report_format == ReportFormat.JSONL:
            return JsonlCodec.encode(result.get("events", []))
        if report_format == ReportFormat.CSV:
            return CsvReporter.render(result)
        if report_format == ReportFormat.TEXT:
            return TextReporter.render(result)
        if report_format == ReportFormat.MARKDOWN:
            return MarkdownReporter.render(result)
        raise ValueError("unsupported report format")


class ResultComparator:
    @staticmethod
    def compare(first, second):
        first_metrics = first.get("metrics", {})
        second_metrics = second.get("metrics", {})
        changes = {}
        for key in sorted(set(first_metrics) | set(second_metrics)):
            left = first_metrics.get(key)
            right = second_metrics.get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                changes[key] = {
                    "before": left,
                    "after": right,
                    "delta": right - left,
                    "relative_change": (right - left) / left if left else None,
                }
            elif left != right:
                changes[key] = {"before": left, "after": right}
        return {
            "first_events": len(first.get("events", [])),
            "second_events": len(second.get("events", [])),
            "changes": changes,
        }


class EventTimeline:
    def __init__(self, events=()):
        self.events = list(events)

    @staticmethod
    def parse_timestamp(value):
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    def ordered(self):
        return sorted(
            self.events,
            key=lambda event: self.parse_timestamp(
                (event.as_dict() if hasattr(event, "as_dict") else event)["timestamp"]
            ),
        )

    def window(self, start, end):
        start = self.parse_timestamp(start)
        end = self.parse_timestamp(end)
        if end < start:
            raise ValueError("window end cannot precede start")
        result = []
        for event in self.ordered():
            data = event.as_dict() if hasattr(event, "as_dict") else event
            timestamp = self.parse_timestamp(data["timestamp"])
            if start <= timestamp <= end:
                result.append(event)
        return result

    def intervals(self):
        ordered = self.ordered()
        timestamps = [
            self.parse_timestamp(
                (event.as_dict() if hasattr(event, "as_dict") else event)["timestamp"]
            )
            for event in ordered
        ]
        return [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:])
        ]


class AtomicStore:
    def __init__(self, root):
        self.root = pathlib.Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise StorageError("path escapes store root")
        return path

    def write(self, name, content):
        path = self._resolve(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, content)
        return path

    def read(self, name):
        return self._resolve(name).read_text(encoding="utf-8")

    def exists(self, name):
        return self._resolve(name).exists()

    def list(self, suffix=None):
        files = sorted(path for path in self.root.rglob("*") if path.is_file())
        if suffix is not None:
            files = [path for path in files if path.suffix == suffix]
        return [str(path.relative_to(self.root)) for path in files]


def atomic_write(path, content):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".chaos-",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ResultArchive:
    def __init__(self, directory):
        self.store = AtomicStore(directory)

    def save(self, name, result):
        if not name or "/" in name or "\\" in name:
            raise StorageError("archive name must be a simple filename")
        return self.store.write(name + ".json", JsonCodec.encode(result) + "\n")

    def load(self, name):
        return JsonCodec.decode(self.store.read(name + ".json"))

    def list(self):
        return self.store.list(".json")

    def compare(self, first, second):
        return ResultComparator.compare(self.load(first), self.load(second))


class ExperimentProfiler:
    def __init__(self):
        self.stages = {}
        self.active = {}

    def begin(self, name):
        if name in self.active:
            raise ExperimentError(f"stage already active: {name}")
        self.active[name] = time.perf_counter()

    def end(self, name):
        if name not in self.active:
            raise ExperimentError(f"stage not active: {name}")
        elapsed = time.perf_counter() - self.active.pop(name)
        self.stages.setdefault(name, []).append(elapsed)
        return elapsed

    def summary(self):
        return {
            name: {
                "calls": len(values),
                "total": sum(values),
                "minimum": min(values),
                "maximum": max(values),
                "mean": statistics.fmean(values),
            }
            for name, values in sorted(self.stages.items())
        }


class ExperimentSession:
    def __init__(self, config, policy=None):
        self.config = config
        self.policy = policy or SafetyPolicy()
        self.machine = StateMachine()
        self.audit = AuditTrail()
        self.sampler = ResourceSampler()
        self.profiler = ExperimentProfiler()
        self.findings = []
        self.result = None

    async def execute(self):
        try:
            self.findings = self.policy.check(self.config)
            if self.findings:
                raise PolicyViolation("experiment failed safety checks")
            self.machine.transition(Lifecycle.VALIDATED)
            self.audit.append("validated", {"duration": self.config.duration})
            self.machine.transition(Lifecycle.READY)
            self.sampler.capture()
            self.machine.transition(Lifecycle.RUNNING)
            self.profiler.begin("experiment")
            try:
                self.result = await run_experiment(self.config)
            finally:
                self.profiler.end("experiment")
            self.machine.transition(Lifecycle.STOPPING)
            self.sampler.capture()
            self.machine.transition(Lifecycle.COMPLETED)
            self.audit.append("completed", {"events": len(self.result["events"])})
            return self.snapshot()
        except asyncio.CancelledError:
            if not self.machine.terminal():
                self.machine.transition(Lifecycle.CANCELLED, "task cancelled")
            self.audit.append("cancelled", {})
            raise
        except Exception as exc:
            if not self.machine.terminal():
                self.machine.transition(Lifecycle.FAILED, type(exc).__name__)
            self.audit.append("failed", {"error": str(exc)})
            raise

    def snapshot(self):
        return {
            "result": self.result,
            "lifecycle": self.machine.snapshot(),
            "audit": self.audit.export(),
            "resources": self.sampler.summary(),
            "profiling": self.profiler.summary(),
            "findings": [finding.as_dict() for finding in self.findings],
        }


class ConfigurationInspector:
    @staticmethod
    def describe(config):
        data = asdict(config)
        data["intensity"] = config.intensity.value
        data["strategies"] = [strategy.value for strategy in config.strategies]
        data["probabilities"] = {
            strategy.value: probability
            for strategy, probability in config.probabilities.items()
        }
        return data

    @staticmethod
    def fingerprint(config):
        encoded = JsonCodec.encode(ConfigurationInspector.describe(config), pretty=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def estimate(config):
        expected = {}
        multiplier = MULTIPLIERS[config.intensity]
        for strategy in config.strategies:
            if strategy == Strategy.RANDOM:
                for candidate in DEFAULT_PROBABILITIES:
                    expected[candidate.value] = expected.get(candidate.value, 0) + (
                        config.requests * min(1.0, config.probabilities[candidate] * multiplier)
                        / len(DEFAULT_PROBABILITIES)
                    )
            else:
                expected[strategy.value] = expected.get(strategy.value, 0) + (
                    config.requests * min(1.0, config.probabilities[strategy] * multiplier)
                )
        return {
            "expected_events_without_deadline_effects": expected,
            "maximum_request_attempts": config.requests * len(config.strategies),
            "maximum_memory_mb": config.max_memory_mb,
            "maximum_cpu_workers": config.max_cpu_workers,
            "experiment_duration_seconds": config.duration,
        }


class HealthEvaluator:
    def __init__(self, minimum_success_rate=0.95, maximum_failure_rate=0.05):
        if not 0 <= minimum_success_rate <= 1:
            raise ValueError("invalid minimum success rate")
        if not 0 <= maximum_failure_rate <= 1:
            raise ValueError("invalid maximum failure rate")
        self.minimum_success_rate = minimum_success_rate
        self.maximum_failure_rate = maximum_failure_rate

    def evaluate(self, payment_metrics):
        total = payment_metrics.get("total", 0)
        if total == 0:
            return {
                "healthy": None,
                "reason": "no payment requests were processed",
                "findings": [],
            }
        statuses = payment_metrics.get("statuses", {})
        accepted = statuses.get(RequestStatus.ACCEPTED.value, 0)
        success_rate = accepted / total
        failure_rate = 1 - success_rate
        findings = []
        if success_rate < self.minimum_success_rate:
            findings.append(Finding(
                "low_acceptance_rate",
                Severity.WARNING,
                "Payment acceptance rate is below target",
                "acceptance_rate",
                success_rate,
                self.minimum_success_rate,
            ))
        if failure_rate > self.maximum_failure_rate:
            findings.append(Finding(
                "high_failure_rate",
                Severity.WARNING,
                "Payment failure rate exceeds target",
                "failure_rate",
                failure_rate,
                self.maximum_failure_rate,
            ))
        return {
            "healthy": not findings,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "findings": [finding.as_dict() for finding in findings],
        }


class ReplayPlan:
    def __init__(self, events):
        self.events = [event.as_dict() if hasattr(event, "as_dict") else dict(event) for event in events]

    def strategies(self):
        return [Strategy(event["strategy"]) for event in self.events]

    def requests(self):
        return list(dict.fromkeys(event["request_id"] for event in self.events))

    def summary(self):
        return {
            "events": len(self.events),
            "requests": len(self.requests()),
            "strategies": dict(Counter(event["strategy"] for event in self.events)),
        }

    def to_scenario(self, name="replay"):
        steps = []
        for index, event in enumerate(self.events, 1):
            steps.append(ScenarioStep(
                name=f"replay_{index:05d}",
                strategy=Strategy(event["strategy"]),
                probability=1.0,
                repeat=1,
            ))
        if not steps:
            raise InvalidScenario("cannot replay an empty event list")
        return Scenario(name=name, steps=tuple(steps), description="Event strategy replay plan")


class StructuredLogger:
    def __init__(self, stream=None):
        import sys
        self.stream = stream if stream is not None else sys.stderr
        self.sequence = 0

    def emit(self, level, message, **fields):
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": str(level).upper(),
            "message": str(message),
            "fields": fields,
        }
        self.stream.write(JsonCodec.encode(record, pretty=False) + "\n")
        return record

    def info(self, message, **fields):
        return self.emit("info", message, **fields)

    def warning(self, message, **fields):
        return self.emit("warning", message, **fields)

    def error(self, message, **fields):
        return self.emit("error", message, **fields)


class ExperimentBatch:
    def __init__(self, configurations, policy=None):
        self.configurations = list(configurations)
        self.policy = policy or SafetyPolicy()
        self.results = []

    async def run(self):
        for index, config in enumerate(self.configurations, 1):
            session = ExperimentSession(config, self.policy)
            outcome = await session.execute()
            self.results.append({"index": index, "outcome": outcome})
        return list(self.results)

    def summary(self):
        totals = Counter()
        for entry in self.results:
            metrics = entry["outcome"]["result"]["metrics"]
            totals["events"] += metrics.get("total_events", 0)
            totals["attempts"] += sum(metrics.get("attempts_by_strategy", {}).values())
        return {
            "completed": len(self.results),
            "requested": len(self.configurations),
            "totals": dict(totals),
        }


class SchemaValidator:
    REQUIRED_EVENT_FIELDS = (
        "event_id",
        "request_id",
        "strategy",
        "status",
        "timestamp",
        "description",
        "duration",
        "component",
        "impact",
    )

    @classmethod
    def validate_event(cls, event):
        data = event.as_dict() if hasattr(event, "as_dict") else event
        findings = []
        if not isinstance(data, dict):
            return [Finding("invalid_event", Severity.ERROR, "Event must be an object")]
        for key in cls.REQUIRED_EVENT_FIELDS:
            if key not in data:
                findings.append(Finding(
                    "missing_event_field",
                    Severity.ERROR,
                    f"Missing event field: {key}",
                    key,
                ))
        if "duration" in data:
            value = data["duration"]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                findings.append(Finding(
                    "invalid_duration",
                    Severity.ERROR,
                    "Event duration must be nonnegative and finite",
                    "duration",
                    value,
                ))
        if "timestamp" in data:
            try:
                datetime.fromisoformat(str(data["timestamp"]).replace("Z", "+00:00"))
            except ValueError:
                findings.append(Finding(
                    "invalid_timestamp",
                    Severity.ERROR,
                    "Event timestamp must be ISO 8601",
                    "timestamp",
                ))
        return findings

    @classmethod
    def validate_result(cls, result):
        findings = []
        if not isinstance(result, dict):
            return [Finding("invalid_result", Severity.ERROR, "Result must be an object")]
        if not isinstance(result.get("events"), list):
            findings.append(Finding("invalid_events", Severity.ERROR, "Events must be an array"))
        else:
            for index, event in enumerate(result["events"]):
                for finding in cls.validate_event(event):
                    findings.append(Finding(
                        finding.code,
                        finding.severity,
                        finding.message,
                        f"events[{index}].{finding.subject}",
                        finding.observed,
                        finding.expected,
                    ))
        if not isinstance(result.get("metrics"), dict):
            findings.append(Finding("invalid_metrics", Severity.ERROR, "Metrics must be an object"))
        return findings


class BaselineTracker:
    def __init__(self, capacity=100):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.history = deque(maxlen=capacity)

    def observe(self, result):
        metrics = result.get("metrics", {})
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_events": metrics.get("total_events", 0),
            "runtime_seconds": metrics.get("runtime_seconds", 0),
            "events_per_minute": metrics.get("events_per_minute", 0),
        }
        self.history.append(snapshot)
        return snapshot

    def average(self, field_name):
        values = [item[field_name] for item in self.history if isinstance(item.get(field_name), (int, float))]
        return statistics.fmean(values) if values else None

    def deviation(self, field_name, value):
        average = self.average(field_name)
        if average is None:
            return None
        return value - average

    def snapshot(self):
        return {
            "count": len(self.history),
            "averages": {
                name: self.average(name)
                for name in ("total_events", "runtime_seconds", "events_per_minute")
            },
        }


class DistributionAnalyzer:
    @staticmethod
    def entropy(counts):
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        return -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
            if count > 0
        )

    @staticmethod
    def gini(counts):
        values = sorted(float(value) for value in counts.values())
        total = sum(values)
        if not values or total <= 0:
            return 0.0
        numerator = sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1))
        return numerator / (len(values) * total)

    @staticmethod
    def proportions(counts):
        total = sum(counts.values())
        return {key: value / total if total else 0.0 for key, value in counts.items()}

    @classmethod
    def analyze(cls, counts):
        return {
            "entropy_bits": cls.entropy(counts),
            "gini": cls.gini(counts),
            "proportions": cls.proportions(counts),
            "total": sum(counts.values()),
        }


class FailureBudget:
    def __init__(self, maximum_failures, window_seconds):
        if maximum_failures < 0 or window_seconds <= 0:
            raise ValueError("invalid failure budget")
        self.maximum_failures = maximum_failures
        self.window_seconds = window_seconds
        self.failures = deque()

    def _expire(self, now):
        cutoff = now - self.window_seconds
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()

    def record(self, now=None):
        now = time.monotonic() if now is None else now
        self._expire(now)
        self.failures.append(now)
        return self.remaining(now)

    def remaining(self, now=None):
        now = time.monotonic() if now is None else now
        self._expire(now)
        return max(0, self.maximum_failures - len(self.failures))

    def exhausted(self, now=None):
        return self.remaining(now) == 0

    def snapshot(self):
        return {
            "maximum_failures": self.maximum_failures,
            "window_seconds": self.window_seconds,
            "current_failures": len(self.failures),
            "remaining": self.remaining(),
        }


class Deadline:
    def __init__(self, duration):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("deadline duration must be positive and finite")
        self.duration = duration
        self.started = time.monotonic()
        self.ends = self.started + duration

    def remaining(self):
        return max(0.0, self.ends - time.monotonic())

    def expired(self):
        return self.remaining() <= 0

    async def sleep(self, duration):
        if duration < 0:
            raise ValueError("sleep duration must be nonnegative")
        actual = min(duration, self.remaining())
        if actual > 0:
            await asyncio.sleep(actual)
        return actual


class RunManifest:
    def __init__(self, config, metadata=None):
        self.config = config
        self.metadata = dict(metadata or {})
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.run_id = uuid.uuid4().hex

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "configuration": ConfigurationInspector.describe(self.config),
            "configuration_sha256": ConfigurationInspector.fingerprint(self.config),
            "metadata": dict(self.metadata),
            "python_version": os.sys.version.split()[0],
            "platform": os.sys.platform,
            "psutil_available": psutil is not None,
        }


class IntegrityVerifier:
    @staticmethod
    def digest(value):
        return hashlib.sha256(JsonCodec.encode(value, pretty=False).encode()).hexdigest()

    @classmethod
    def sign_result(cls, result):
        return {
            "algorithm": "sha256",
            "digest": cls.digest(result),
            "result": result,
        }

    @classmethod
    def verify_result(cls, envelope):
        if envelope.get("algorithm") != "sha256":
            raise VerificationError("unsupported digest algorithm")
        expected = cls.digest(envelope.get("result"))
        if not hmac.compare_digest(expected, envelope.get("digest", "")):
            raise VerificationError("result integrity check failed")
        return True


class ExperimentCatalog:
    def __init__(self):
        self.entries = {}

    def register(self, name, description, factory):
        if not name or not callable(factory):
            raise ValueError("catalog entry requires name and factory")
        if name in self.entries:
            raise ValueError("duplicate catalog entry")
        self.entries[name] = {"description": description, "factory": factory}

    def create(self, name):
        if name not in self.entries:
            raise KeyError(name)
        return self.entries[name]["factory"]()

    def describe(self):
        return {
            name: entry["description"]
            for name, entry in sorted(self.entries.items())
        }


def built_in_scenarios():
    return (
        Scenario(
            "gateway_recovery",
            (
                ScenarioStep("outage", Strategy.GATEWAY_OUTAGE),
                ScenarioStep("latency", Strategy.LATENCY, repeat=2),
                ScenarioStep("failure", Strategy.FAILURE),
            ),
            "Simulated gateway degradation and recovery",
            ("payment", "availability"),
        ),
        Scenario(
            "service_partition",
            (
                ScenarioStep("partition", Strategy.NETWORK_PARTITION),
                ScenarioStep("retry_delay", Strategy.LATENCY, repeat=3),
                ScenarioStep("retry_failure", Strategy.FAILURE),
            ),
            "Simulated downstream service isolation",
            ("network", "resilience"),
        ),
        Scenario(
            "resource_pressure",
            (
                ScenarioStep("cpu", Strategy.CPU_SPIKE),
                ScenarioStep("memory", Strategy.MEMORY_LEAK),
                ScenarioStep("response", Strategy.LATENCY),
            ),
            "Bounded CPU and memory pressure",
            ("resources", "capacity"),
        ),
        Scenario(
            "mixed_resilience",
            (
                ScenarioStep("random_faults", Strategy.RANDOM, repeat=5),
                ScenarioStep("latency", Strategy.LATENCY, repeat=2),
                ScenarioStep("gateway", Strategy.GATEWAY_OUTAGE),
            ),
            "Mixed fault experiment",
            ("mixed", "simulation"),
        ),
    )


def create_catalog():
    catalog = ExperimentCatalog()
    for scenario in built_in_scenarios():
        catalog.register(
            scenario.name,
            scenario.description,
            lambda scenario=scenario: scenario,
        )
    return catalog


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    if args.list_scenarios:
        print(JsonCodec.encode(create_catalog().describe()))
        return 0
    if args.verify_result:
        try:
            envelope = json.loads(pathlib.Path(args.verify_result).read_text(encoding="utf-8"))
            IntegrityVerifier.verify_result(envelope)
        except (OSError, ValueError, VerificationError) as exc:
            parser.error(str(exc))
        print("Result integrity verified")
        return 0
    probabilities = dict(DEFAULT_PROBABILITIES)
    probabilities.update(args.probability)
    try:
        config = Config(
            duration=args.duration,
            requests=args.requests,
            request_interval=args.request_interval,
            intensity=Intensity(args.intensity),
            strategies=tuple(dict.fromkeys(Strategy(x) for x in args.strategies)),
            max_latency=args.max_latency,
            max_memory_mb=args.max_memory_mb,
            max_cpu_duration=args.max_cpu_duration,
            gateway_outage_duration=args.gateway_outage_duration,
            network_partition_duration=args.network_partition_duration,
            max_concurrency=args.max_concurrency,
            max_cpu_workers=args.max_cpu_workers,
            history_limit=args.history_limit,
            memory_headroom_mb=args.memory_headroom_mb,
            seed=args.seed,
            dry_run=args.dry_run,
            probabilities=probabilities,
        )
        SafetyPolicy().enforce(config)
    except (ValueError, PolicyViolation) as exc:
        parser.error(str(exc))
    if args.inspect:
        print(JsonCodec.encode({
            "configuration": ConfigurationInspector.describe(config),
            "estimate": ConfigurationInspector.estimate(config),
            "fingerprint": ConfigurationInspector.fingerprint(config),
        }))
        return 0

    async def execute():
        if args.scenario or args.scenario_file:
            if args.scenario_file:
                scenario = Scenario.from_dict(
                    json.loads(pathlib.Path(args.scenario_file).read_text(encoding="utf-8"))
                )
            else:
                scenario = create_catalog().create(args.scenario)
            injector = ChaosInjector(config)
            runner = ScenarioRunner(injector, seed=config.seed)
            try:
                injector.start()
                events = await runner.execute(scenario)
                return {
                    "events": [event.as_dict() for event in events],
                    "metrics": injector.metrics(),
                    "scenario": scenario.as_dict(),
                }
            finally:
                injector.close()
        if args.audit:
            session = ExperimentSession(config)
            return await session.execute()
        return await run_experiment(config)

    try:
        result = asyncio.run(execute())
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, InvalidScenario, PolicyViolation) as exc:
        parser.error(str(exc))
    if args.archive_directory:
        archive = ResultArchive(args.archive_directory)
        archive.save(args.archive_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"), result)
    if args.json:
        report = JsonCodec.encode(result) + "\n"
    else:
        report = ReportFactory.render(result, args.report_format)
    if args.output:
        atomic_write(args.output, report)
        print(f"Report saved to {args.output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
