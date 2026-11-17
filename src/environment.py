"""Simulador del cruce: dos calles (A y B) que comparten un semáforo.

Idea general:
- Las llegadas de autos se generan ANTES de que el semáforo decida nada,
  así todos los controladores enfrentan exactamente el mismo tránsito.
- Cada 5 segundos el controlador elige: 0 = mantener la fase, 1 = pedir un cambio.
- Cada cola guarda, para cada auto, el segundo en que llegó (para medir su espera).
"""
from dataclasses import dataclass
import numpy as np

SCENARIOS = ("equilibrado", "desigual", "cambiante", "estres")


@dataclass
class Config:
    """Parámetros del cruce (en segundos, salvo la capacidad)."""
    horizon: int = 600             # duración de un episodio
    decision_seconds: int = 5      # cada cuánto decide el controlador
    min_green: int = 10            # verde mínimo
    max_green: int = 60            # verde máximo
    clearance: int = 3             # transición (amarillo) en la que nadie cruza
    service_per_second: int = 1    # autos que cruzan por segundo con verde


def arrivals_for(scenario, seed, horizon):
    """Autos que llegan a cada calle en cada segundo (matriz horizon x 2).

    Se sortean con una distribución de Poisson usando las tasas del escenario
    (autos por segundo para la calle A y la calle B).
    """
    rates = np.zeros((horizon, 2), dtype=float)
    if scenario == "equilibrado":
        rates[:] = (.23, .23)
    elif scenario == "desigual":
        rates[:] = (.40, .10)
    elif scenario == "cambiante":
        # La primera mitad está cargada la calle A y la segunda mitad la calle B.
        rates[:horizon // 2] = (.40, .10)
        rates[horizon // 2:] = (.10, .40)
    elif scenario == "estres":
        rates[:] = (.42, .32)
    else:
        raise ValueError(f"Escenario desconocido: {scenario}")
    return np.random.default_rng(seed).poisson(rates)


class TrafficEnv:
    """Entorno de aprendizaje por refuerzo.

    reset() empieza un episodio y devuelve la primera observación.
    step(accion) avanza 5 segundos y devuelve (observación, recompensa, terminó).
    """

    def __init__(self, config=None):
        if config is None:
            config = Config()
        self.config = config

    def reset(self, scenario="equilibrado", seed=0, arrivals=None):
        c = self.config
        if arrivals is None:
            self.arrivals = arrivals_for(scenario, seed, c.horizon)
        else:
            self.arrivals = np.array(arrivals, dtype=int)  # llegadas de Montevideo
        self.queues = [[], []]     # cola de la calle A y cola de la calle B
        # La fase inicial se sortea con otra semilla para no tocar las llegadas.
        self.phase = int(np.random.default_rng(seed + 9_000_000).integers(2))
        self.age = 0               # segundos que lleva el verde actual
        self.clear_remaining = 0   # segundos de transición que faltan
        self.t = 0                 # segundo actual del episodio
        self.switches = 0          # cantidad de cambios de fase
        self.wait_area = 0         # suma de autos esperando, segundo a segundo
        self.departed_waits = []   # espera de cada auto que ya cruzó
        self.max_queue = 0
        self.trace = []            # historial segundo a segundo (para gráficos y demo)
        return self.observation()

    def observation(self):
        """Lo que ve el controlador, con valores llevados a una escala cercana a 0-1."""
        return np.array([len(self.queues[0]) / 40,             # cola de A
                         len(self.queues[1]) / 40,             # cola de B
                         self.phase,                           # 0: verde A, 1: verde B
                         self.age / self.config.max_green,     # edad del verde
                         self.t / self.config.horizon],        # tiempo transcurrido
                        dtype=np.float32)

    def step(self, action, record=False):
        c = self.config

        # 1) Decidir si cambia la fase.
        #    Cambia si se llegó al verde máximo, o si se pidió y ya pasó el verde mínimo.
        if self.clear_remaining == 0:
            llega_al_maximo = self.age + c.decision_seconds > c.max_green
            cambio_permitido = action == 1 and self.age >= c.min_green
            if llega_al_maximo or cambio_permitido:
                self.phase = 1 - self.phase
                self.age = 0
                self.clear_remaining = c.clearance
                self.switches += 1

        # 2) Simular los próximos 5 segundos, uno por uno.
        area = 0  # autos esperando sumados en estos segundos
        segundos = min(c.decision_seconds, c.horizon - self.t)
        for _ in range(segundos):
            # Llegan autos: se anota el segundo de llegada de cada uno.
            for calle in (0, 1):
                nuevos = int(self.arrivals[self.t, calle])
                self.queues[calle].extend([self.t] * nuevos)

            if self.clear_remaining > 0:
                # Transición: nadie cruza.
                active = -1
                self.clear_remaining -= 1
            else:
                # Verde: cruzan autos de la calle habilitada, el primero que llegó primero.
                active = self.phase
                cola = self.queues[self.phase]
                for _ in range(c.service_per_second):
                    if cola:
                        llegada = cola.pop(0)
                        self.departed_waits.append(self.t - llegada)
                self.age += 1

            esperando = len(self.queues[0]) + len(self.queues[1])
            area += esperando
            self.max_queue = max(self.max_queue, esperando)
            if record:
                self.trace.append(dict(t=self.t, ns=len(self.queues[0]),
                                       eo=len(self.queues[1]), active=active,
                                       wait_area=self.wait_area + area,
                                       departed=len(self.departed_waits)))
            self.t += 1
        self.wait_area += area

        # 3) Recompensa: menos los autos esperando (dividido entre 100 para que
        #    los números no sean tan grandes al entrenar la red de DQN).
        reward = -area / 100.
        done = self.t >= c.horizon
        return self.observation(), reward, done

    def metrics(self):
        """Resumen del episodio. Los autos que siguen en cola cuentan con lo que llevan esperando."""
        arrived = int(self.arrivals[:self.t].sum())
        pending_waits = [self.t - llegada for cola in self.queues for llegada in cola]
        observed = self.departed_waits + pending_waits
        p95 = float(np.percentile(observed, 95)) if observed else 0.
        mean_departed = float(np.mean(self.departed_waits)) if self.departed_waits else 0.
        return dict(arrived=arrived,                                 # autos que llegaron
                    departed=len(self.departed_waits),               # autos que cruzaron
                    pending=len(pending_waits),                      # autos que quedaron en cola
                    wait_area=self.wait_area,                        # segundos de espera en total
                    mean_observed_wait=self.wait_area / max(arrived, 1),  # espera media por auto
                    p95_observed_wait=p95,                           # espera del 5 % que más esperó
                    mean_departed_wait=mean_departed,
                    max_queue=self.max_queue,
                    switches=self.switches,
                    ns_pending=len(self.queues[0]),
                    eo_pending=len(self.queues[1]))
