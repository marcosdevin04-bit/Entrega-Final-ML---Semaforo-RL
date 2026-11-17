"""Selecciona el semáforo fijo en validación y compara políticas sin exploración."""
import argparse
import json
import numpy as np
from src.controllers import FixedController
from src.environment import SCENARIOS
from src.data import load_windows, real_arrivals
from src.experiment import ROOT, read_json, write_csv, load_controller, run_episode

def select_baseline(cfg):
    windows = load_windows('validation')
    rows = []
    for green in cfg['fixed_green_candidates']:
        for i in range(20):
            seed = cfg['validation_seed_start'] + i
            for scenario in ('equilibrado', 'desigual', 'cambiante', 'montevideo'):
                arrivals = None
                if scenario == 'montevideo':
                    window = windows[(i * 23) % len(windows)][1]
                    arrivals = real_arrivals(window, seed)
                m, _ = run_episode(FixedController(green),
                                   'equilibrado' if scenario == 'montevideo' else scenario, seed, arrivals)
                rows.append(dict(green=green, scenario=scenario, seed=seed,
                                 mean_observed_wait=m['mean_observed_wait']))
    means = {g:float(np.mean([r['mean_observed_wait'] for r in rows if r['green']==g]))
             for g in cfg['fixed_green_candidates']}
    selected = min(means, key=means.get)
    write_csv(ROOT / 'results' / 'baseline_validation.csv', rows)
    (ROOT / 'results' / 'baseline_selection.json').write_text(
        json.dumps(dict(selected_green=selected, validation_means=means), indent=2), encoding='utf-8')
    print('Verde fijo elegido únicamente en validación:', selected, flush=True)

def evaluate(cfg, n):
    windows = load_windows('test')
    # Muestreo reproducible de ventanas de fechas reservadas. Pueden solaparse:
    # no se interpretan como 30 días independientes de observación real.
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(windows), size=n, replace=n > len(windows))
    rows, episode_rows = [], []
    controllers = {name: {s:load_controller(name,s) for s in
                   (cfg['train_seeds'] if name in ('q_learning','dqn') else [-1])}
                   for name in ('fijo','reglas','q_learning','dqn')}
    for scenario in (*SCENARIOS, 'montevideo'):
        for i in range(n):
            seed = cfg['evaluation_seed_start'] + i
            arrivals = None
            timestamp = ''
            if scenario == 'montevideo':
                timestamp, window = windows[idx[i]]
                arrivals = real_arrivals(window, seed)
            episode_rows.append(dict(scenario=scenario, traffic_seed=seed, window_start_endstamp=timestamp))
            for algorithm in ('fijo', 'reglas', 'q_learning', 'dqn'):
                for train_seed in (cfg['train_seeds'] if algorithm in ('q_learning', 'dqn') else [-1]):
                    controller = controllers[algorithm][train_seed]
                    m, _ = run_episode(controller, 'equilibrado' if scenario == 'montevideo' else scenario,
                                       seed, arrivals)
                    rows.append(dict(scenario=scenario, algorithm=algorithm,
                                     train_seed=train_seed, traffic_seed=seed, **m))
        print('Evaluado:', scenario, flush=True)
    write_csv(ROOT / 'results' / 'evaluation.csv', rows)
    write_csv(ROOT / 'results' / 'evaluation_windows.csv', episode_rows)
    summary = []
    for scenario in (*SCENARIOS, 'montevideo'):
        for algorithm in ('fijo','reglas','q_learning','dqn'):
            subset = [r for r in rows if r['scenario']==scenario and r['algorithm']==algorithm]
            row = dict(scenario=scenario, algorithm=algorithm, n_runs=len(subset))
            for metric in ('mean_observed_wait','p95_observed_wait','pending','departed','max_queue','switches'):
                values = [r[metric] for r in subset]
                row[metric] = float(np.mean(values))
                row[metric + '_std'] = float(np.std(values, ddof=1))
            summary.append(row)
    write_csv(ROOT / 'results' / 'summary.csv', summary)
    # Diferencias pareadas por semilla de tránsito. Primero promediar los 5 agentes.
    # Así no contamos una misma realización de tránsito como 5 observaciones independientes.
    paired = []
    for scenario in (*SCENARIOS, 'montevideo'):
        base = {r['traffic_seed']:r['mean_observed_wait'] for r in rows
                if r['scenario']==scenario and r['algorithm']=='fijo'}
        for algorithm in ('reglas','q_learning','dqn'):
            diffs = []
            for seed in base:
                scores = [r['mean_observed_wait'] for r in rows if r['scenario']==scenario
                          and r['algorithm']==algorithm and r['traffic_seed']==seed]
                diffs.append(float(np.mean(scores))-base[seed])
            boot = np.random.default_rng(42).choice(diffs, (2000, len(diffs)), replace=True).mean(axis=1)
            paired.append(dict(scenario=scenario, algorithm=algorithm,
                               difference_seconds=float(np.mean(diffs)),
                               ci95_low=float(np.percentile(boot,2.5)), ci95_high=float(np.percentile(boot,97.5))))
    write_csv(ROOT / 'results' / 'paired_differences.csv', paired)
    print('Resultados guardados en results/', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--select-baseline', action='store_true')
    p.add_argument('--episodes', type=int)
    args = p.parse_args()
    cfg = read_json(ROOT / 'configs' / 'experiment.json')
    (ROOT / 'results').mkdir(exist_ok=True)
    if args.select_baseline or not (ROOT / 'results' / 'baseline_selection.json').exists():
        select_baseline(cfg)
    n = args.episodes if args.episodes is not None else cfg['evaluation_episodes']
    if n <= 0:
        p.error('episodes debe ser positivo')
    evaluate(cfg, n)

if __name__ == '__main__':
    main()
