"""DQN en PyTorch: replay uniforme, red objetivo y pérdida Huber.

No implementa Double DQN, dueling ni replay priorizado. La red recibe las cinco
observaciones continuas del simulador; no usa la discretización de Q-learning.
"""
import numpy as np
import torch
from torch import nn

class DQN:
    def __init__(self, seed=0, gamma=.97, lr=.0005, capacity=30000):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.online = nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        self.target = nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.capacity, self.size, self.position, self.updates = capacity, 0, 0, 0
        self.states = np.empty((capacity, 5), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def act(self, obs, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        with torch.no_grad():
            return int(self.online(torch.from_numpy(np.asarray(obs, dtype=np.float32))).argmax().item())

    def remember(self, obs, action, reward, nxt, done):
        p = self.position
        self.states[p], self.actions[p], self.rewards[p] = obs, action, reward
        self.next_states[p], self.dones[p] = nxt, done
        self.position = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def learn(self, batch_size=64):
        if self.size < 1000:
            return None
        idx = self.rng.choice(self.size, batch_size, replace=False)
        s, ns = torch.from_numpy(self.states[idx]), torch.from_numpy(self.next_states[idx])
        a = torch.from_numpy(self.actions[idx])
        r, done = torch.from_numpy(self.rewards[idx]), torch.from_numpy(self.dones[idx])
        with torch.no_grad():
            targets = r + self.gamma * (1 - done) * self.target(ns).max(1).values
        predictions = self.online(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 250 == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        # Solo pesos de inferencia. No es un checkpoint para reanudar entrenamiento.
        torch.save(self.online.state_dict(), path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.online.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        obj.target.load_state_dict(obj.online.state_dict())
        return obj
