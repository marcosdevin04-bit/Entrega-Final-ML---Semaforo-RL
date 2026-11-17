"""Elige el semáforo fijo en validación y compara los cuatro controladores en prueba.

Uso:
    python -m scripts.evaluate                    # solo la prueba final
    python -m scripts.evaluate --select-baseline  # además vuelve a elegir el fijo
"""
import argparse
import json
import numpy as np
import pandas as pd
from src.controllers import FixedController
from src.environment import SCENARIOS
from src.data import load_windows, real_arrivals
from src.experiment import ROOT, read_json, load_controller, run_episode

RESULTS = ROOT / 'results'
ESCENARIOS = (*SCENARIOS, 'montevideo')
CONTROLADORES = ('fijo', 'reglas', 'q_learning', 'dqn')
METRICAS = ('mean_observed_wait', 'p95_observed_wait', 'pending',
            'departed', 'max_queue', 'switches')


def select_baseline(cfg):
    """Prueba cada duración de verde en VALIDACIÓN y se queda con la de menor espera media."""
    windows = load_windows('validation')
    filas = []
    for green in cfg['fixed_green_candidates']:
        for i in range(20):
            seed = cfg['validation_seed_start'] + i
            for escenario in ('equilibrado', 'desigual', 'cambiante', 'montevideo'):
                if escenario == 'montevideo':
                    window = windows[(i * 23) % len(windows)][1]
                    m, _ = run_episode(FixedController(green), 'equilibrado', seed,
                                       real_arrivals(window, seed))
                else:
                    m, _ = run_episode(FixedController(green), escenario, seed)
                filas.append(dict(green=green, scenario=escenario, seed=seed,
                                  mean_observed_wait=m['mean_observed_wait']))

    df = pd.DataFrame(filas)
    df.to_csv(RESULTS / 'baseline_validation.csv', index=False)
    medias = df.groupby('green')['mean_observed_wait'].apply(np.mean)
    elegido = int(medias.idxmin())
    resumen = dict(selected_green=elegido,
                   validation_means={int(g): float(v) for g, v in medias.items()})
    (RESULTS / 'baseline_selection.json').write_text(json.dumps(resumen, indent=2), encoding='utf-8')
    print('Verde fijo elegido únicamente en validación:', elegido, flush=True)


def evaluate(cfg, n):
    # 1) Elegir n ventanas de Montevideo de las fechas de PRUEBA (26 al 31 de enero).
    #    Pueden solaparse: no se interpretan como n días independientes.
    windows = load_windows('test')
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(windows), size=n, replace=n > len(windows))

    # 2) Cargar los controladores. Cada agente tiene 5 versiones (una por semilla de entrenamiento).
    controladores = {}
    for nombre in CONTROLADORES:
        semillas = cfg['train_seeds'] if nombre in ('q_learning', 'dqn') else [-1]
        controladores[nombre] = {s: load_controller(nombre, s) for s in semillas}

    # 3) Correr todos los episodios. Para cada escenario y semilla de tránsito,
    #    los cuatro controladores enfrentan exactamente las mismas llegadas.
    filas, ventanas = [], []
    for escenario in ESCENARIOS:
        for i in range(n):
            seed = cfg['evaluation_seed_start'] + i
            arrivals, timestamp = None, ''
            if escenario == 'montevideo':
                timestamp, window = windows[idx[i]]
                arrivals = real_arrivals(window, seed)
            ventanas.append(dict(scenario=escenario, traffic_seed=seed,
                                 window_start_endstamp=timestamp))
            entorno = 'equilibrado' if escenario == 'montevideo' else escenario
            for nombre in CONTROLADORES:
                for train_seed, controlador in controladores[nombre].items():
                    m, _ = run_episode(controlador, entorno, seed, arrivals)
                    filas.append(dict(scenario=escenario, algorithm=nombre,
                                      train_seed=train_seed, traffic_seed=seed, **m))
        print('Evaluado:', escenario, flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(RESULTS / 'evaluation.csv', index=False)
    pd.DataFrame(ventanas).to_csv(RESULTS / 'evaluation_windows.csv', index=False)

    # 4) Tabla resumen: media y desvío de cada métrica por escenario y controlador.
    resumen = []
    for escenario in ESCENARIOS:
        for nombre in CONTROLADORES:
            sub = df[(df['scenario'] == escenario) & (df['algorithm'] == nombre)]
            fila = dict(scenario=escenario, algorithm=nombre, n_runs=len(sub))
            for metrica in METRICAS:
                fila[metrica] = float(np.mean(sub[metrica]))
                fila[metrica + '_std'] = float(np.std(sub[metrica], ddof=1))
            resumen.append(fila)
    pd.DataFrame(resumen).to_csv(RESULTS / 'summary.csv', index=False)

    # 5) Diferencia de cada controlador contra el fijo, semilla por semilla (comparación pareada).
    #    Primero se promedian los 5 agentes de una misma semilla de tránsito, para no contar
    #    el mismo tránsito como 5 observaciones independientes. Después, bootstrap de 2000 remuestreos.
    pareadas = []
    for escenario in ESCENARIOS:
        del_escenario = df[df['scenario'] == escenario]
        fijo = del_escenario[del_escenario['algorithm'] == 'fijo']
        fijo = fijo.set_index('traffic_seed')['mean_observed_wait']
        for nombre in ('reglas', 'q_learning', 'dqn'):
            agente = del_escenario[del_escenario['algorithm'] == nombre]
            agente = agente.groupby('traffic_seed')['mean_observed_wait'].apply(np.mean)
            diffs = (agente - fijo).to_numpy()   # negativo = espera menos que el fijo
            muestras = np.random.default_rng(42).choice(diffs, (2000, len(diffs)), replace=True)
            medias_boot = muestras.mean(axis=1)
            pareadas.append(dict(scenario=escenario, algorithm=nombre,
                                 difference_seconds=float(np.mean(diffs)),
                                 ci95_low=float(np.percentile(medias_boot, 2.5)),
                                 ci95_high=float(np.percentile(medias_boot, 97.5))))
    pd.DataFrame(pareadas).to_csv(RESULTS / 'paired_differences.csv', index=False)
    print('Resultados guardados en results/', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--select-baseline', action='store_true')
    p.add_argument('--episodes', type=int)
    args = p.parse_args()
    cfg = read_json(ROOT / 'configs' / 'experiment.json')
    RESULTS.mkdir(exist_ok=True)
    if args.select_baseline or not (RESULTS / 'baseline_selection.json').exists():
        select_baseline(cfg)
    n = args.episodes if args.episodes is not None else cfg['evaluation_episodes']
    evaluate(cfg, n)


if __name__ == '__main__':
    main()
