"""DQN (Deep Q-Network) en PyTorch.

Es la misma idea que Q-learning, pero en vez de una tabla se usa una red
neuronal: recibe las 5 observaciones (sin agrupar en intervalos) y devuelve
dos números, el valor Q de mantener y el valor Q de cambiar.

Agrega dos piezas para que el entrenamiento sea estable:
- Memoria de experiencias (replay): se guardan las jugadas y se aprende de
  lotes mezclados al azar, en vez de solo de la última jugada.
- Red objetivo: una copia de la red que se actualiza cada tanto y se usa
  para calcular el objetivo, así el objetivo no cambia en cada paso.
"""
import numpy as np
import torch
from torch import nn


def crear_red():
    # 5 entradas -> 64 -> 64 -> 2 salidas (una por acción), con ReLU como en clase.
    return nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 2))


class DQN:
    def __init__(self, semilla=0, gamma=.97, tasa_aprendizaje=.0005, capacidad=30000):
        torch.set_num_threads(1)
        torch.manual_seed(semilla)
        self.rng = np.random.default_rng(semilla)
        self.gamma = gamma

        self.red = crear_red()           # la red que aprende
        self.red_objetivo = crear_red()  # la copia que se usa para el objetivo
        self.red_objetivo.load_state_dict(self.red.state_dict())
        self.optimizador = torch.optim.Adam(self.red.parameters(), lr=tasa_aprendizaje)

        # Memoria de experiencias: arreglos de tamaño fijo que se van llenando.
        # Cuando se llena, las jugadas nuevas pisan a las más viejas.
        self.capacidad = capacidad
        self.guardadas = 0       # cuántas jugadas hay guardadas
        self.posicion = 0        # dónde se guarda la próxima
        self.entrenamientos = 0  # cuántas veces se entrenó la red
        self.estados = np.empty((capacidad, 5), dtype=np.float32)
        self.siguientes = np.empty((capacidad, 5), dtype=np.float32)
        self.acciones = np.empty(capacidad, dtype=np.int64)
        self.recompensas = np.empty(capacidad, dtype=np.float32)
        self.terminados = np.empty(capacidad, dtype=np.float32)

    def actuar(self, obs, epsilon=0.):
        """Epsilon-greedy: con probabilidad epsilon explora al azar; si no, la acción de mayor Q."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            valores_q = self.red(x)
            return int(valores_q.argmax().item())

    def recordar(self, obs, accion, recompensa, siguiente, termino):
        """Guarda una jugada (estado, acción, recompensa, estado siguiente, terminó)."""
        p = self.posicion
        self.estados[p] = obs
        self.acciones[p] = accion
        self.recompensas[p] = recompensa
        self.siguientes[p] = siguiente
        self.terminados[p] = termino
        self.posicion = (p + 1) % self.capacidad
        self.guardadas = min(self.guardadas + 1, self.capacidad)

    def aprender(self, tamano_lote=64):
        """Un paso de entrenamiento con un lote de 64 jugadas tomadas al azar de la memoria."""
        if self.guardadas < 1000:
            return None   # se espera a tener al menos 1000 jugadas guardadas

        # 1) Elegir el lote al azar.
        idx = self.rng.choice(self.guardadas, tamano_lote, replace=False)
        s = torch.from_numpy(self.estados[idx])
        a = torch.from_numpy(self.acciones[idx])
        r = torch.from_numpy(self.recompensas[idx])
        s_sig = torch.from_numpy(self.siguientes[idx])
        termino = torch.from_numpy(self.terminados[idx])

        # 2) Objetivo, igual que en Q-learning: r + gamma * max Q(s', a').
        #    Se calcula con la red objetivo y sin gradiente. Si terminó, solo r.
        with torch.no_grad():
            mejor_siguiente = self.red_objetivo(s_sig).max(1).values
            objetivo = r + self.gamma * (1 - termino) * mejor_siguiente

        # 3) Predicción: de cada fila, el valor Q de la acción que se tomó.
        filas = torch.arange(tamano_lote)
        prediccion = self.red(s)[filas, a]

        # 4) Pérdida Huber: como el error cuadrático, pero los errores grandes pesan menos.
        perdida = nn.functional.smooth_l1_loss(prediccion, objetivo)

        # 5) Paso de gradiente, igual que en el lab de redes.
        self.optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red.parameters(), 10.)  # evita pasos demasiado grandes
        self.optimizador.step()

        # 6) Cada 250 entrenamientos se copia la red que aprende en la red objetivo.
        self.entrenamientos += 1
        if self.entrenamientos % 250 == 0:
            self.red_objetivo.load_state_dict(self.red.state_dict())
        return float(perdida.item())

    def guardar(self, ruta):
        # Solo se guardan los pesos de la red (alcanza para usarla, no para seguir entrenando).
        torch.save(self.red.state_dict(), ruta)

    @classmethod
    def cargar(cls, ruta):
        agente = cls()
        agente.red.load_state_dict(torch.load(ruta, map_location='cpu', weights_only=True))
        agente.red_objetivo.load_state_dict(agente.red.state_dict())
        return agente
