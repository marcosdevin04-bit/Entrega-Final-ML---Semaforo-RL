"""Controles de referencia sometidos a las restricciones del entorno."""
class FixedController:
    def __init__(self, green=25):
        self.green = green

    def act(self, obs):
        return int(round(float(obs[3]) * 60) >= self.green)

class QueueController:
    def act(self, obs):
        phase = int(obs[2])
        return int(obs[1 - phase] > obs[phase])
