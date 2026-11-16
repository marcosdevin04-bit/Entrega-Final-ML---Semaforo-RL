"""Entrena Q-learning y/o DQN con el mismo presupuesto de interacción."""
import argparse
import time
import json
import platform
import numpy as np
from src.environment import TrafficEnv
from src.q_learning import QLearning
from src.data import load_windows, real_arrivals
from src.experiment import ROOT, read_json, write_csv

def train(algorithm, seed, cfg, episodes):
    if algorithm == 'q_learning':
        agent = QLearning(seed, cfg['q_alpha'], cfg['gamma'])
    else:
        from src.dqn import DQN
        agent = DQN(seed, cfg['gamma'], cfg['dqn_learning_rate'])
    env = TrafficEnv()
    windows = load_windows('train')
    # Generador independiente: ambos algoritmos ven los mismos episodios.
    rng = np.random.default_rng(10000 + seed)
    rows, step = [], 0
    started = time.perf_counter()
    for episode in range(episodes):
        scenario = ('equilibrado', 'desigual', 'cambiante', 'montevideo')[int(rng.integers(4))]
        arrival_seed = 100000 + seed * 10000 + episode
        arrivals = None
        if scenario == 'montevideo':
            _, window = windows[int(rng.integers(len(windows)))]
            arrivals = real_arrivals(window, arrival_seed)
        obs = env.reset('equilibrado' if scenario == 'montevideo' else scenario, arrival_seed, arrivals)
        fraction = min(1., episode / max(1, episodes * cfg['exploration_fraction']))
        epsilon = cfg['epsilon_initial'] + fraction * (cfg['epsilon_final'] - cfg['epsilon_initial'])
        total_reward, losses, done = 0., [], False
        while not done:
            action = agent.act(obs, epsilon)
            nxt, reward, done = env.step(action)
            if algorithm == 'q_learning':
                agent.update(obs, action, reward, nxt, done)
            else:
                agent.remember(obs, action, reward, nxt, done)
                if step % cfg['dqn_train_frequency'] == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
            total_reward += reward
            obs = nxt
            step += 1
        rows.append(dict(algorithm=algorithm, train_seed=seed, episode=episode + 1,
                         scenario=scenario, epsilon=epsilon, reward=total_reward,
                         mean_loss=float(np.mean(losses)) if losses else '',
                         mean_observed_wait=env.metrics()['mean_observed_wait']))
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            print(f'{algorithm} semilla={seed} episodio={episode+1}/{episodes} '
                  f'tiempo={time.perf_counter()-started:.1f}s', flush=True)
    (ROOT / 'models').mkdir(exist_ok=True)
    extension = 'npz' if algorithm == 'q_learning' else 'pt'
    agent.save(ROOT / 'models' / f'{algorithm}_{seed}.{extension}')
    write_csv(ROOT / 'results' / f'training_{algorithm}_{seed}.csv', rows)
    metadata = dict(algorithm=algorithm, seed=seed, episodes=episodes, decisions=step,
                    seconds=time.perf_counter()-started, config=cfg,
                    python=platform.python_version(), numpy=np.__version__, device='cpu')
    if algorithm == 'dqn':
        import torch
        metadata['torch'] = torch.__version__
    (ROOT / 'models' / f'{algorithm}_{seed}.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['q_learning', 'dqn', 'all'], default='all')
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    cfg = read_json(ROOT / 'configs' / 'experiment.json')
    episodes = cfg['episodes'] if args.episodes is None else args.episodes
    if episodes <= 0:
        parser.error('episodes debe ser positivo')
    for algorithm in (['q_learning', 'dqn'] if args.algorithm == 'all' else [args.algorithm]):
        for seed in (args.seeds if args.seeds is not None else cfg['train_seeds']):
            train(algorithm, seed, cfg, episodes)

if __name__ == '__main__':
    main()
