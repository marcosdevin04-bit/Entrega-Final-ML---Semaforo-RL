"""Funciones compartidas para entrenamiento, evaluación y demostración."""
from pathlib import Path
import csv
import json
from .environment import TrafficEnv
from .q_learning import QLearning
from .controllers import FixedController, QueueController

ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def load_controller(name, seed=0):
    if name == 'fijo':
        return FixedController(read_json(ROOT / 'results' / 'baseline_selection.json')['selected_green'])
    if name == 'reglas':
        return QueueController()
    if name == 'q_learning':
        return QLearning.load(ROOT / 'models' / f'q_learning_{seed}.npz')
    if name == 'dqn':
        from .dqn import DQN
        return DQN.load(ROOT / 'models' / f'dqn_{seed}.pt')
    raise ValueError('Controlador desconocido: ' + name)

def run_episode(controller, scenario, seed, arrivals=None, record=False):
    env = TrafficEnv()
    obs = env.reset(scenario, seed, arrivals)
    done = False
    while not done:
        obs, _, done = env.step(controller.act(obs), record=record)
    return env.metrics(), env.trace
