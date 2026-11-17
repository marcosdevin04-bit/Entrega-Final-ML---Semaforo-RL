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
    def __init__(self, seed=0, gamma=.97, lr=.0005, capacity=30000):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma

        self.online = crear_red()   # la red que aprende
        self.target = crear_red()   # la red objetivo (copia)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        # Memoria de experiencias: arreglos de tamaño fijo que se van llenando.
        # Cuando se llena, las jugadas nuevas pisan a las más viejas.
        self.capacity = capacity
        self.size = 0        # cuántas jugadas hay guardadas
        self.position = 0    # dónde se guarda la próxima
        self.updates = 0     # cuántas veces se entrenó la red
        self.states = np.empty((capacity, 5), dtype=np.float32)
        self.next_states = np.empty((capacity, 5), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def act(self, obs, epsilon=0.):
        """Epsilon-greedy: con probabilidad epsilon explora al azar; si no, la acción de mayor Q."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            valores_q = self.online(x)
            return int(valores_q.argmax().item())

    def remember(self, obs, action, reward, nxt, done):
        """Guarda una jugada (estado, acción, recompensa, estado siguiente, terminó)."""
        p = self.position
        self.states[p] = obs
        self.actions[p] = action
        self.rewards[p] = reward
        self.next_states[p] = nxt
        self.dones[p] = done
        self.position = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def learn(self, batch_size=64):
        """Un paso de entrenamiento con un lote de 64 jugadas tomadas al azar de la memoria."""
        if self.size < 1000:
            return None   # se espera a tener al menos 1000 jugadas guardadas

        # 1) Elegir el lote al azar.
        idx = self.rng.choice(self.size, batch_size, replace=False)
        s = torch.from_numpy(self.states[idx])
        a = torch.from_numpy(self.actions[idx])
        r = torch.from_numpy(self.rewards[idx])
        s_sig = torch.from_numpy(self.next_states[idx])
        done = torch.from_numpy(self.dones[idx])

        # 2) Objetivo, igual que en Q-learning: r + gamma * max Q(s', a').
        #    Se calcula con la red objetivo y sin gradiente. Si terminó, solo r.
        with torch.no_grad():
            mejor_siguiente = self.target(s_sig).max(1).values
            objetivo = r + self.gamma * (1 - done) * mejor_siguiente

        # 3) Predicción: de cada fila, el valor Q de la acción que se tomó.
        filas = torch.arange(batch_size)
        prediccion = self.online(s)[filas, a]

        # 4) Pérdida Huber: como el error cuadrático, pero los errores grandes pesan menos.
        loss = nn.functional.smooth_l1_loss(prediccion, objetivo)

        # 5) Paso de gradiente, igual que en el lab de redes.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)  # evita pasos demasiado grandes
        self.optimizer.step()

        # 6) Cada 250 entrenamientos se copia la red que aprende en la red objetivo.
        self.updates += 1
        if self.updates % 250 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        # Solo se guardan los pesos de la red (alcanza para usarla, no para seguir entrenando).
        torch.save(self.online.state_dict(), path)

    @classmethod
    def load(cls, path):
        agente = cls()
        agente.online.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        agente.target.load_state_dict(agente.online.state_dict())
        return agente
