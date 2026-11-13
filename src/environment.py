"""Simulador discreto. Las llegadas se generan ANTES de elegir acciones."""
from collections import deque
from dataclasses import dataclass
import numpy as np

SCENARIOS = ("equilibrado", "desigual", "cambiante", "estres")

@dataclass(frozen=True)
class Config:
    horizon: int = 600
    decision_seconds: int = 5
    min_green: int = 10
    max_green: int = 60
    clearance: int = 3
    service_per_second: int = 1

    def __post_init__(self):
        if not (self.horizon > 0 and self.decision_seconds > 0 and
                0 < self.min_green <= self.max_green and
                0 <= self.clearance <= self.decision_seconds and
                self.service_per_second > 0):
            raise ValueError("Configuración temporal o capacidad inválida")

def arrivals_for(scenario, seed, horizon):
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {scenario}")
    rates = np.zeros((horizon, 2), dtype=float)
    if scenario == "equilibrado":
        rates[:] = (.23, .23)
    elif scenario == "desigual":
        rates[:] = (.40, .10)
    elif scenario == "cambiante":
        rates[:horizon // 2] = (.40, .10)
        rates[horizon // 2:] = (.10, .40)
    else:
        rates[:] = (.42, .32)
    return np.random.default_rng(seed).poisson(rates)

class TrafficEnv:
    """API pequeña propia: reset devuelve observación; step devuelve obs,r,done.

    Un paso de decisión incluye hasta cinco segundos. En cada segundo:
    llegan vehículos, salen los habilitados, y quienes quedan suman 1 s de espera.
    La fase inicial se sortea con un generador independiente de las llegadas.
    """
    def __init__(self, config=None):
        self.config = config or Config()

    def reset(self, scenario="equilibrado", seed=0, arrivals=None):
        c = self.config
        self.arrivals = (arrivals_for(scenario, seed, c.horizon) if arrivals is None
                         else np.array(arrivals, dtype=int, copy=True))
        if self.arrivals.shape != (c.horizon, 2) or np.any(self.arrivals < 0):
            raise ValueError("Las llegadas deben tener forma (horizon, 2) y ser no negativas")
        self.queues = [deque(), deque()]
        self.phase = int(np.random.default_rng(seed + 9_000_000).integers(2))
        self.age = 0
        self.clear_remaining = 0
        self.t = 0
        self.switches = 0
        self.wait_area = 0
        self.departed_waits = []
        self.max_queue = 0
        self.trace = []
        return self.observation()

    def observation(self):
        return np.array([len(self.queues[0]) / 40, len(self.queues[1]) / 40,
                         self.phase, self.age / self.config.max_green,
                         self.t / self.config.horizon], dtype=np.float32)

    def step(self, action, record=False):
        if self.t >= self.config.horizon:
            raise RuntimeError("Episodio finalizado: llame a reset")
        if action not in (0, 1):
            raise ValueError("Acciones válidas: 0 mantener, 1 solicitar cambio")
        c = self.config
        if not self.clear_remaining and (self.age + c.decision_seconds > c.max_green or
                (action == 1 and self.age >= c.min_green)):
            self.phase = 1 - self.phase
            self.age = 0
            self.clear_remaining = c.clearance
            self.switches += 1
        area = 0
        for _ in range(min(c.decision_seconds, c.horizon - self.t)):
            for direction in (0, 1):
                self.queues[direction].extend([self.t] * int(self.arrivals[self.t, direction]))
            active = -1 if self.clear_remaining else self.phase
            if self.clear_remaining:
                self.clear_remaining -= 1
            else:
                q = self.queues[self.phase]
                for _ in range(min(c.service_per_second, len(q))):
                    self.departed_waits.append(self.t - q.popleft())
                self.age += 1
            total = len(self.queues[0]) + len(self.queues[1])
            area += total
            self.max_queue = max(self.max_queue, total)
            if record:
                self.trace.append(dict(t=self.t, ns=len(self.queues[0]),
                                       eo=len(self.queues[1]), active=active,
                                       wait_area=self.wait_area + area,
                                       departed=len(self.departed_waits)))
            self.t += 1
        self.wait_area += area
        # Escala constante: conserva el objetivo y estabiliza las magnitudes de DQN.
        return self.observation(), -area / 100., self.t >= c.horizon

    def metrics(self):
        arrived = int(self.arrivals[:self.t].sum())
        pending_waits = [self.t - a for q in self.queues for a in q]
        observed = self.departed_waits + pending_waits
        # Las identidades de conservación se comprueban en tests/test_project.py.
        return dict(arrived=arrived, departed=len(self.departed_waits),
                    pending=len(pending_waits), wait_area=self.wait_area,
                    mean_observed_wait=self.wait_area / max(arrived, 1),
                    p95_observed_wait=float(np.percentile(observed, 95)) if observed else 0.,
                    mean_departed_wait=float(np.mean(self.departed_waits)) if self.departed_waits else 0.,
                    max_queue=self.max_queue, switches=self.switches,
                    ns_pending=len(self.queues[0]), eo_pending=len(self.queues[1]))
