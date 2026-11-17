"""Entrena Q-learning y/o DQN con el mismo presupuesto de interacción.

Uso:
    python -m scripts.entrenar                                   # los 10 agentes
    python -m scripts.entrenar --algoritmo q_learning --episodios 100 --semillas 0

Ojo: reemplaza los modelos de models/ y los registros de results/.
"""
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
from src.entorno import Cruce
from src.q_learning import QLearning
from src.datos import cargar_ventanas, llegadas_montevideo
from src.experimento import RAIZ, leer_json


def entrenar(algoritmo, semilla, cfg, episodios):
    if algoritmo == 'q_learning':
        agente = QLearning(semilla, cfg['alpha_q_learning'], cfg['gamma'])
    else:
        from src.dqn import DQN
        agente = DQN(semilla, cfg['gamma'], cfg['tasa_aprendizaje_dqn'])
    cruce = Cruce()
    ventanas = cargar_ventanas('entrenamiento')
    # Generador propio para elegir escenarios: con la misma semilla, los dos
    # algoritmos ven exactamente la misma secuencia de episodios.
    rng = np.random.default_rng(10000 + semilla)
    filas, paso = [], 0
    inicio = time.perf_counter()

    for episodio in range(episodios):
        # 1) Elegir el escenario del episodio (los cuatro con igual probabilidad).
        escenario = ('equilibrado', 'desigual', 'cambiante', 'montevideo')[int(rng.integers(4))]
        semilla_llegadas = 100000 + semilla * 10000 + episodio
        llegadas = None
        if escenario == 'montevideo':
            _, tasas = ventanas[int(rng.integers(len(ventanas)))]
            llegadas = llegadas_montevideo(tasas, semilla_llegadas)
        obs = cruce.reiniciar(escenario, semilla_llegadas, llegadas)

        # 2) Exploración: epsilon baja en línea recta de 1 a 0,05 durante el primer 75 %.
        avance = min(1., episodio / max(1, episodios * cfg['fraccion_exploracion']))
        epsilon = cfg['epsilon_inicial'] + avance * (cfg['epsilon_final'] - cfg['epsilon_inicial'])

        # 3) Jugar el episodio y aprender de cada decisión.
        recompensa_total, perdidas, termino = 0., [], False
        while not termino:
            accion = agente.actuar(obs, epsilon)
            siguiente, recompensa, termino = cruce.avanzar(accion)
            if algoritmo == 'q_learning':
                agente.actualizar(obs, accion, recompensa, siguiente, termino)
            else:
                agente.recordar(obs, accion, recompensa, siguiente, termino)
                if paso % cfg['frecuencia_entrenamiento_dqn'] == 0:
                    perdida = agente.aprender()
                    if perdida is not None:
                        perdidas.append(perdida)
            recompensa_total += recompensa
            obs = siguiente
            paso += 1

        filas.append(dict(algoritmo=algoritmo, semilla=semilla, episodio=episodio + 1,
                          escenario=escenario, epsilon=epsilon, recompensa=recompensa_total,
                          perdida_media=float(np.mean(perdidas)) if perdidas else None,
                          espera_media=cruce.metricas()['espera_media']))
        if (episodio + 1) % 100 == 0 or episodio == episodios - 1:
            print(f'{algoritmo} semilla={semilla} episodio={episodio + 1}/{episodios} '
                  f'tiempo={time.perf_counter() - inicio:.1f}s', flush=True)

    # 4) Guardar el modelo, el registro del entrenamiento y una ficha con versiones y tiempo.
    (RAIZ / 'models').mkdir(exist_ok=True)
    extension = 'npz' if algoritmo == 'q_learning' else 'pt'
    agente.guardar(RAIZ / 'models' / f'{algoritmo}_{semilla}.{extension}')
    pd.DataFrame(filas).to_csv(RAIZ / 'results' / f'entrenamiento_{algoritmo}_{semilla}.csv', index=False)
    ficha = dict(algoritmo=algoritmo, semilla=semilla, episodios=episodios, decisiones=paso,
                 segundos=time.perf_counter() - inicio, configuracion=cfg,
                 python=platform.python_version(), numpy=np.__version__, dispositivo='cpu')
    if algoritmo == 'dqn':
        import torch
        ficha['torch'] = torch.__version__
    (RAIZ / 'models' / f'{algoritmo}_{semilla}.json').write_text(
        json.dumps(ficha, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--algoritmo', choices=['q_learning', 'dqn', 'todos'], default='todos')
    parser.add_argument('--episodios', type=int)
    parser.add_argument('--semillas', nargs='+', type=int)
    args = parser.parse_args()
    cfg = leer_json(RAIZ / 'configs' / 'experimento.json')
    episodios = cfg['episodios'] if args.episodios is None else args.episodios
    algoritmos = ['q_learning', 'dqn'] if args.algoritmo == 'todos' else [args.algoritmo]
    semillas = args.semillas if args.semillas is not None else cfg['semillas_entrenamiento']
    (RAIZ / 'results').mkdir(exist_ok=True)
    for algoritmo in algoritmos:
        for semilla in semillas:
            entrenar(algoritmo, semilla, cfg, episodios)


if __name__ == '__main__':
    main()
