"""Comprobaciones puntuales del simulador y los algoritmos, fuera del flujo principal."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.environment import TrafficEnv, Config, SCENARIOS, arrivals_for
from src.q_learning import QLearning, encode
from src.data import load_windows, real_arrivals

class SimulationTests(unittest.TestCase):
    def test_conservation_and_waits(self):
        for scenario in SCENARIOS:
            env = TrafficEnv()
            env.reset(scenario, 42)
            done = False
            while not done:
                _, _, done = env.step(1)
            m = env.metrics()
            remaining_wait = sum(env.t-a for q in env.queues for a in q)
            self.assertEqual(m['arrived'], m['departed'] + m['pending'])
            self.assertEqual(m['wait_area'], sum(env.departed_waits) + remaining_wait)

    def test_clearance_and_minimum_green(self):
        env = TrafficEnv(Config(horizon=40))
        arrivals = np.ones((40, 2), dtype=int)
        env.reset(seed=3, arrivals=arrivals)
        initial = env.phase
        env.step(1)
        env.step(1)
        self.assertEqual(env.phase, initial)
        before = len(env.departed_waits)
        env.step(1, record=True)
        self.assertEqual(env.phase, 1-initial)
        self.assertEqual([r['active'] for r in env.trace[:3]], [-1, -1, -1])
        self.assertEqual(len(env.departed_waits)-before, 2)

    def test_maximum_green(self):
        env = TrafficEnv()
        env.reset(seed=1)
        done = False
        while not done:
            _, _, done = env.step(0)
            self.assertLessEqual(env.age, env.config.max_green)
        self.assertGreater(env.switches, 0)

    def test_demand_does_not_depend_on_actions(self):
        a, b = TrafficEnv(), TrafficEnv()
        a.reset('cambiante', 8)
        b.reset('cambiante', 8)
        for _ in range(10):
            a.step(0)
            b.step(1)
        np.testing.assert_array_equal(a.arrivals, b.arrivals)

    def test_zero_arrivals_and_partial_last_step(self):
        env = TrafficEnv(Config(horizon=13))
        env.reset(arrivals=np.zeros((13, 2)))
        done, reward = False, 0
        while not done:
            _, r, done = env.step(0)
            reward += r
        self.assertEqual(env.t, 13)
        self.assertEqual(reward, 0)
        self.assertEqual(env.metrics()['mean_observed_wait'], 0)

    def test_temporal_data_split(self):
        for split, low, high in [('train', 1, 20), ('validation', 21, 25), ('test', 26, 31)]:
            for stamp, rates in load_windows(split):
                self.assertTrue(low <= int(stamp[8:10]) <= high)
                self.assertTrue(np.all(rates >= 0))
            arrivals = real_arrivals(load_windows(split)[0][1], 42)
            self.assertEqual(arrivals.shape, (600, 2))

class LearningTests(unittest.TestCase):
    def test_terminal_q_update_and_save(self):
        agent = QLearning(alpha=1)
        obs = np.zeros(5, dtype=np.float32)
        agent.q[:] = 100
        agent.update(obs, 1, -3, obs, True)
        self.assertEqual(agent.q[encode(obs)+(1,)], -3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'q.npz'
            agent.save(path)
            np.testing.assert_array_equal(agent.q, QLearning.load(path).q)

    def test_dqn_replay_training_and_target(self):
        import torch
        from src.dqn import DQN
        agent = DQN(seed=0)
        obs = np.ones(5, dtype=np.float32)
        for _ in range(1000):
            agent.remember(obs, 0, -1, obs, True)
        target_before = [p.detach().clone() for p in agent.target.parameters()]
        online_before = [p.detach().clone() for p in agent.online.parameters()]
        loss = agent.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(online_before,agent.online.parameters())))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(target_before,agent.target.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'dqn.pt'
            agent.save(path)
            restored = DQN.load(path)
            with torch.no_grad():
                np.testing.assert_allclose(agent.online(torch.from_numpy(obs)),
                                           restored.online(torch.from_numpy(obs)))

if __name__ == '__main__':
    unittest.main()
