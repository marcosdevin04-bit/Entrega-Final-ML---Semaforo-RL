"""Q-learning tabular con agregación explícita de observaciones."""
import numpy as np

QUEUE_EDGES = np.array([0, 2, 5, 10, 20, 40])
AGE_EDGES = np.array([9, 19, 29, 39, 49, 59])
SHAPE = (7, 7, 2, 7, 3, 2)

def encode(obs):
    return (int(np.digitize(round(float(obs[0]) * 40), QUEUE_EDGES, right=True)),
            int(np.digitize(round(float(obs[1]) * 40), QUEUE_EDGES, right=True)),
            int(obs[2]), int(np.digitize(round(float(obs[3]) * 60), AGE_EDGES, right=True)),
            min(2, int(float(obs[4]) * 3)))

class QLearning:
    def __init__(self, seed=0, alpha=.12, gamma=.97):
        self.q = np.zeros(SHAPE, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma

    def act(self, obs, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        return int(np.argmax(self.q[encode(obs)]))

    def update(self, obs, action, reward, nxt, done):
        key = encode(obs) + (action,)
        target = reward + (0 if done else self.gamma * float(self.q[encode(nxt)].max()))
        self.q[key] += self.alpha * (target - self.q[key])

    def save(self, path):
        np.savez_compressed(path, q=self.q)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as data:
            obj.q = data['q'].copy()
        return obj
