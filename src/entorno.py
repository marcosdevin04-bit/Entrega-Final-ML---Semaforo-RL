"""Simulador del cruce: dos calles (A y B) que comparten un semáforo.

Idea general:
- Las llegadas de autos se generan ANTES de que el semáforo decida nada,
  así todos los controladores enfrentan exactamente el mismo tránsito.
- Cada 5 segundos el controlador elige: 0 = mantener la fase, 1 = pedir un cambio.
- Cada cola guarda, para cada auto, el segundo en que llegó (para medir su espera).

Los métodos `reiniciar` y `avanzar` cumplen el papel de `reset` y `step` en
bibliotecas de aprendizaje por refuerzo como Gymnasium.
"""
from dataclasses import dataclass
import numpy as np

ESCENARIOS = ("equilibrado", "desigual", "cambiante", "estres")


@dataclass
class Configuracion:
    """Parámetros del cruce (en segundos, salvo la capacidad)."""
    duracion: int = 600            # duración de un episodio
    segundos_decision: int = 5     # cada cuánto decide el controlador
    verde_minimo: int = 10
    verde_maximo: int = 60
    transicion: int = 3            # amarillo: nadie cruza
    autos_por_segundo: int = 1     # autos que cruzan por segundo con verde


def generar_llegadas(escenario, semilla, duracion):
    """Autos que llegan a cada calle en cada segundo (matriz duracion x 2).

    Se sortean con una distribución de Poisson usando las tasas del escenario
    (autos por segundo para la calle A y la calle B).
    """
    tasas = np.zeros((duracion, 2), dtype=float)
    if escenario == "equilibrado":
        tasas[:] = (.23, .23)
    elif escenario == "desigual":
        tasas[:] = (.40, .10)
    elif escenario == "cambiante":
        # La primera mitad está cargada la calle A y la segunda mitad la calle B.
        tasas[:duracion // 2] = (.40, .10)
        tasas[duracion // 2:] = (.10, .40)
    elif escenario == "estres":
        tasas[:] = (.42, .32)
    else:
        raise ValueError(f"Escenario desconocido: {escenario}")
    return np.random.default_rng(semilla).poisson(tasas)


class Cruce:
    """Entorno de aprendizaje por refuerzo.

    reiniciar() empieza un episodio y devuelve la primera observación.
    avanzar(accion) simula 5 segundos y devuelve (observación, recompensa, terminó).
    """

    def __init__(self, config=None):
        if config is None:
            config = Configuracion()
        self.config = config

    def reiniciar(self, escenario="equilibrado", semilla=0, llegadas=None):
        c = self.config
        if llegadas is None:
            self.llegadas = generar_llegadas(escenario, semilla, c.duracion)
        else:
            self.llegadas = np.array(llegadas, dtype=int)  # llegadas de Montevideo
        self.colas = [[], []]      # cola de la calle A y cola de la calle B
        # La fase inicial se sortea con otra semilla para no tocar las llegadas.
        self.fase = int(np.random.default_rng(semilla + 9_000_000).integers(2))
        self.edad_verde = 0        # segundos que lleva el verde actual
        self.transicion_restante = 0
        self.t = 0                 # segundo actual del episodio
        self.cambios = 0           # cantidad de cambios de fase
        self.espera_total = 0      # suma de autos esperando, segundo a segundo
        self.esperas_cruzaron = [] # espera de cada auto que ya cruzó
        self.cola_maxima = 0
        self.historial = []        # registro segundo a segundo (para gráficos y demo)
        return self.observacion()

    def observacion(self):
        """Lo que ve el controlador, con valores llevados a una escala cercana a 0-1."""
        return np.array([len(self.colas[0]) / 40,               # cola de A
                         len(self.colas[1]) / 40,               # cola de B
                         self.fase,                             # 0: verde A, 1: verde B
                         self.edad_verde / self.config.verde_maximo,
                         self.t / self.config.duracion],        # tiempo transcurrido
                        dtype=np.float32)

    def avanzar(self, accion, registrar=False):
        c = self.config

        # 1) Decidir si cambia la fase.
        #    Cambia si se llegó al verde máximo, o si se pidió y ya pasó el verde mínimo.
        if self.transicion_restante == 0:
            llega_al_maximo = self.edad_verde + c.segundos_decision > c.verde_maximo
            cambio_permitido = accion == 1 and self.edad_verde >= c.verde_minimo
            if llega_al_maximo or cambio_permitido:
                self.fase = 1 - self.fase
                self.edad_verde = 0
                self.transicion_restante = c.transicion
                self.cambios += 1

        # 2) Simular los próximos 5 segundos, uno por uno.
        area = 0  # autos esperando sumados en estos segundos
        segundos = min(c.segundos_decision, c.duracion - self.t)
        for _ in range(segundos):
            # Llegan autos: se anota el segundo de llegada de cada uno.
            for calle in (0, 1):
                nuevos = int(self.llegadas[self.t, calle])
                self.colas[calle].extend([self.t] * nuevos)

            if self.transicion_restante > 0:
                # Transición: nadie cruza.
                fase_activa = -1
                self.transicion_restante -= 1
            else:
                # Verde: cruzan autos de la calle habilitada, el primero que llegó primero.
                fase_activa = self.fase
                cola = self.colas[self.fase]
                for _ in range(c.autos_por_segundo):
                    if cola:
                        llegada = cola.pop(0)
                        self.esperas_cruzaron.append(self.t - llegada)
                self.edad_verde += 1

            esperando = len(self.colas[0]) + len(self.colas[1])
            area += esperando
            self.cola_maxima = max(self.cola_maxima, esperando)
            if registrar:
                self.historial.append(dict(t=self.t, cola_a=len(self.colas[0]),
                                           cola_b=len(self.colas[1]), fase_activa=fase_activa,
                                           espera_total=self.espera_total + area,
                                           cruzaron=len(self.esperas_cruzaron)))
            self.t += 1
        self.espera_total += area

        # 3) Recompensa: menos los autos esperando (dividido entre 100 para que
        #    los números no sean tan grandes al entrenar la red de DQN).
        recompensa = -area / 100.
        termino = self.t >= c.duracion
        return self.observacion(), recompensa, termino

    def metricas(self):
        """Resumen del episodio. Los autos que siguen en cola cuentan con lo que llevan esperando."""
        llegaron = int(self.llegadas[:self.t].sum())
        esperas_pendientes = [self.t - llegada for cola in self.colas for llegada in cola]
        observadas = self.esperas_cruzaron + esperas_pendientes
        p95 = float(np.percentile(observadas, 95)) if observadas else 0.
        media_cruzaron = float(np.mean(self.esperas_cruzaron)) if self.esperas_cruzaron else 0.
        return dict(llegaron=llegaron,                               # autos que llegaron
                    cruzaron=len(self.esperas_cruzaron),             # autos que cruzaron
                    pendientes=len(esperas_pendientes),              # autos que quedaron en cola
                    espera_total=self.espera_total,                  # segundos de espera en total
                    espera_media=self.espera_total / max(llegaron, 1),  # espera media por auto
                    espera_p95=p95,                                  # espera del 5 % que más esperó
                    espera_media_cruzaron=media_cruzaron,
                    cola_maxima=self.cola_maxima,
                    cambios=self.cambios,
                    pendientes_a=len(self.colas[0]),
                    pendientes_b=len(self.colas[1]))
