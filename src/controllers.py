"""Controles de referencia: no aprenden nada, siguen una regla fija.

Los dos reciben la misma observación que los agentes:
obs = [cola_a / 40, cola_b / 40, fase, edad_verde / 60, tiempo / 600]
y devuelven 0 (mantener la fase) o 1 (pedir un cambio).
Las reglas del cruce (verde mínimo, máximo y transición) las aplica el entorno.
"""
MANTENER = 0
CAMBIAR = 1


class FixedController:
    """Semáforo de tiempos fijos: pide cambiar cuando el verde llega a `green` segundos."""

    def __init__(self, green=25):
        self.green = green

    def act(self, obs):
        edad_verde = round(float(obs[3]) * 60)  # segundos que lleva el verde actual
        if edad_verde >= self.green:
            return CAMBIAR
        return MANTENER


class QueueController:
    """Regla de mayor cola: pide cambiar si la otra calle tiene más autos esperando."""

    def act(self, obs):
        fase = int(obs[2])        # 0: verde para A, 1: verde para B
        cola_actual = obs[fase]   # la calle que tiene verde
        cola_otra = obs[1 - fase] # la calle que está en rojo
        if cola_otra > cola_actual:
            return CAMBIAR
        return MANTENER
