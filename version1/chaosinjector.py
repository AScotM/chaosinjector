#!/usr/bin/env python3

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

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

    def to_dict(self) -> Dict[str, Any]:
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
    def __init__(self):
        self.events: List[ChaosEvent] = []
        self._event_count = 0
        self._start_time = time.time()
    
    def record_event(self, event: ChaosEvent):
        self.events.append(event)
        self._event_count += 1
        logger.info(f"Chaos event recorded: {event.description}")
    
    def get_metrics(self) -> Dict[str, Any]:
        current_time = time.time()
        runtime = current_time - self._start_time
        
        events_by_strategy = {}
        for strategy in ChaosStrategy:
            events_by_strategy[strategy.value] = len([
                e for e in self.events if e.strategy == strategy
            ])
        
        return {
            "total_events": self._event_count,
            "runtime_seconds": runtime,
            "events_per_minute": self._event_count / (runtime / 60) if runtime > 0 else 0,
            "events_by_strategy": events_by_strategy,
            "recent_events": [e.to_dict() for e in self.events[-10:]]
        }
    
    def clear_events(self):
        self.events.clear()
        self._event_count = 0
        self._start_time = time.time()

class MemoryLeakSimulator:
    def __init__(self, max_leak_mb: int = 100):
        self.max_leak_mb = max_leak_mb
        self._leaked_data = []
    
    async def inject_memory_leak(self, size_mb: int) -> ChaosEvent:
        leak_size = min(size_mb, self.max_leak_mb)
        
        if leak_size > 0:
            chunk_size = 1024 * 1024
            data = bytearray()
            for i in range(leak_size):
                data.extend(b'X' * chunk_size)
                await asyncio.sleep(0.001)
            
            self._leaked_data.append(data)
        
        return ChaosEvent(
            event_id=f"memleak_{int(time.time())}",
            strategy=ChaosStrategy.MEMORY_LEAK,
            intensity=ChaosIntensity.MEDIUM,
            timestamp=datetime.now(),
            description=f"Injected memory leak of {leak_size}MB",
            affected_component="memory",
            duration=0.0,
            impact=f"Allocated {leak_size}MB of memory"
        )
    
    def cleanup(self):
        self._leaked_data.clear()

class CPUSpikeSimulator:
    async def inject_cpu_spike(self, duration: float) -> ChaosEvent:
        start_time = time.time()
        
        while time.time() - start_time < duration:
            _ = [x * x for x in range(10000)]
            await asyncio.sleep(0.001)
        
        return ChaosEvent(
            event_id=f"cpu_{int(time.time())}",
            strategy=ChaosStrategy.CPU_SPIKE,
            intensity=ChaosIntensity.MEDIUM,
            timestamp=datetime.now(),
            description=f"Injected CPU spike for {duration:.2f}s",
            affected_component="cpu",
            duration=duration,
            impact=f"CPU intensive computation for {duration:.2f}s"
        )

class ChaosInjector:
    def __init__(self, config: Optional[ChaosConfig] = None):
        self.config = config or ChaosConfig()
        self.config.__post_init__()
        self.monitor = ChaosMonitor()
        self.memory_leak_simulator = MemoryLeakSimulator(self.config.max_memory_leak_mb)
        self.cpu_spike_simulator = CPUSpikeSimulator()
        
        self.gateway_outages: Dict[str, float] = {}
        self.network_partitions: Dict[str, float] = {}
        self._active_chaos = False
    
    async def inject_payment_chaos(self, payment_id: str) -> List[ChaosEvent]:
        if not self._active_chaos:
            return []
        
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
    
    async def _apply_strategy(self, strategy: ChaosStrategy, payment_id: str) -> Optional[ChaosEvent]:
        try:
            if strategy == ChaosStrategy.LATENCY and random.random() < self.config.latency_probability:
                return await self._inject_latency(payment_id)
            
            elif strategy == ChaosStrategy.FAILURE and random.random() < self.config.failure_probability:
                return await self._inject_failure(payment_id)
            
            elif strategy == ChaosStrategy.GATEWAY_OUTAGE and random.random() < self.config.gateway_outage_probability:
                return await self._inject_gateway_outage(payment_id)
            
            elif strategy == ChaosStrategy.MEMORY_LEAK and random.random() < self.config.memory_leak_probability:
                return await self._inject_memory_leak(payment_id)
            
            elif strategy == ChaosStrategy.CPU_SPIKE and random.random() < self.config.cpu_spike_probability:
                return await self._inject_cpu_spike(payment_id)
            
            elif strategy == ChaosStrategy.NETWORK_PARTITION and random.random() < self.config.network_partition_probability:
                return await self._inject_network_partition(payment_id)
            
            elif strategy == ChaosStrategy.RANDOM:
                available_strategies = [s for s in ChaosStrategy if s != ChaosStrategy.RANDOM]
                random_strategy = random.choice(available_strategies)
                return await self._apply_strategy(random_strategy, payment_id)
        
        except Exception as e:
            logger.error(f"Error applying chaos strategy {strategy}: {e}")
        
        return None
    
    async def _inject_latency(self, payment_id: str) -> ChaosEvent:
        latency = random.uniform(0.1, self.config.max_latency)
        await asyncio.sleep(latency)
        
        return ChaosEvent(
            event_id=f"latency_{int(time.time())}",
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
            event_id=f"failure_{int(time.time())}",
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
        outage_end = time.time() + self.config.gateway_outage_duration
        
        self.gateway_outages[affected_gateway] = outage_end
        
        return ChaosEvent(
            event_id=f"outage_{int(time.time())}",
            strategy=ChaosStrategy.GATEWAY_OUTAGE,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Simulated outage for {affected_gateway} gateway",
            affected_component=f"gateway_{affected_gateway}",
            duration=self.config.gateway_outage_duration,
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
        partition_end = time.time() + self.config.network_partition_duration
        self.network_partitions[affected_component] = partition_end
        
        return ChaosEvent(
            event_id=f"partition_{int(time.time())}",
            strategy=ChaosStrategy.NETWORK_PARTITION,
            intensity=self.config.intensity,
            timestamp=datetime.now(),
            description=f"Simulated network partition for {affected_component}",
            affected_component=affected_component,
            duration=self.config.network_partition_duration,
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
    
    def cleanup_expired_chaos(self):
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
    
    def enable_chaos(self):
        self._active_chaos = True
        logger.info("Chaos injection enabled")
    
    def disable_chaos(self):
        self._active_chaos = False
        logger.info("Chaos injection disabled")
    
    def set_intensity(self, intensity: ChaosIntensity):
        self.config.intensity = intensity
        self.config.__post_init__()
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
            "enabled_strategies": [s.value for s in self.config.enabled_strategies]
        }
        
        return {**base_metrics, **current_chaos}
    
    async def run_chaos_experiment(self, duration: int = 300):
        logger.info(f"Starting chaos experiment for {duration} seconds")
        self.enable_chaos()
        
        start_time = time.time()
        experiment_events = []
        
        try:
            while time.time() - start_time < duration:
                await asyncio.sleep(random.uniform(10, 30))
                
                if self._active_chaos:
                    events = await self.inject_payment_chaos("experiment")
                    experiment_events.extend(events)
        
        finally:
            self.disable_chaos()
            logger.info(f"Chaos experiment completed. Injected {len(experiment_events)} events")
        
        return experiment_events

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
    
    return ChaosInjector(config)

async def demo_chaos_injector():
    injector = create_chaos_injector(ChaosIntensity.MEDIUM)
    injector.enable_chaos()
    
    print("Chaos Injector Demo")
    print("=" * 50)
    
    for i in range(5):
        events = await injector.inject_payment_chaos(f"demo_payment_{i}")
        for event in events:
            print(f"Chaos Event: {event.description}")
        
        await asyncio.sleep(1)
    
    metrics = injector.get_chaos_metrics()
    print(f"\nChaos Metrics: {json.dumps(metrics, indent=2)}")
    
    injector.disable_chaos()

if __name__ == "__main__":
    asyncio.run(demo_chaos_injector())
