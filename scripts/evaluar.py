"""Elige el semáforo fijo en validación y compara los cuatro controladores en prueba.

Uso:
    python -m scripts.evaluar               # solo la prueba final
    python -m scripts.evaluar --elegir-fijo # además vuelve a elegir el verde del fijo
"""
import argparse
import json
import numpy as np
import pandas as pd
from src.controladores import SemaforoFijo
from src.entorno import ESCENARIOS
from src.datos import cargar_ventanas, llegadas_montevideo
from src.experimento import RAIZ, CONTROLADORES, leer_json, cargar_controlador, correr_episodio

RESULTADOS = RAIZ / 'results'
ESCENARIOS_PRUEBA = (*ESCENARIOS, 'montevideo')
METRICAS = ('espera_media', 'espera_p95', 'pendientes', 'cruzaron', 'cola_maxima', 'cambios')


def elegir_fijo(cfg):
    """Prueba cada duración de verde en VALIDACIÓN y se queda con la de menor espera media."""
    ventanas = cargar_ventanas('validacion')
    filas = []
    for verde in cfg['verdes_fijos_candidatos']:
        for i in range(20):
            semilla = cfg['semilla_validacion_inicial'] + i
            for escenario in ('equilibrado', 'desigual', 'cambiante', 'montevideo'):
                llegadas = None
                if escenario == 'montevideo':
                    tasas = ventanas[(i * 23) % len(ventanas)][1]
                    llegadas = llegadas_montevideo(tasas, semilla)
                m, _ = correr_episodio(SemaforoFijo(verde), escenario, semilla, llegadas)
                filas.append(dict(verde=verde, escenario=escenario, semilla=semilla,
                                  espera_media=m['espera_media']))

    df = pd.DataFrame(filas)
    df.to_csv(RESULTADOS / 'validacion_fijo.csv', index=False)
    medias = df.groupby('verde')['espera_media'].apply(np.mean)
    elegido = int(medias.idxmin())
    seleccion = dict(verde_elegido=elegido,
                     medias_validacion={int(v): float(m) for v, m in medias.items()})
    (RESULTADOS / 'seleccion_fijo.json').write_text(json.dumps(seleccion, indent=2), encoding='utf-8')
    print('Verde fijo elegido en validación:', elegido, flush=True)


def evaluar(cfg, n):
    # 1) Elegir n ventanas de Montevideo de las fechas de PRUEBA (26 al 31 de enero).
    #    Pueden solaparse: no se interpretan como n días independientes.
    ventanas = cargar_ventanas('prueba')
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(ventanas), size=n, replace=n > len(ventanas))

    # 2) Cargar los controladores. Cada agente tiene 5 versiones (una por semilla de entrenamiento).
    #    Las reglas no se entrenan: se les asigna la semilla -1.
    controladores = {}
    for nombre in CONTROLADORES:
        semillas = cfg['semillas_entrenamiento'] if nombre in ('q_learning', 'dqn') else [-1]
        controladores[nombre] = {s: cargar_controlador(nombre, s) for s in semillas}

    # 3) Correr todos los episodios. Para cada escenario y semilla de tránsito,
    #    los cuatro controladores enfrentan exactamente las mismas llegadas.
    filas, ventanas_usadas = [], []
    for escenario in ESCENARIOS_PRUEBA:
        for i in range(n):
            semilla = cfg['semilla_evaluacion_inicial'] + i
            llegadas, momento = None, ''
            if escenario == 'montevideo':
                momento, tasas = ventanas[idx[i]]
                llegadas = llegadas_montevideo(tasas, semilla)
            ventanas_usadas.append(dict(escenario=escenario, semilla_transito=semilla,
                                        ventana=momento))
            for nombre in CONTROLADORES:
                for semilla_entrenamiento, controlador in controladores[nombre].items():
                    m, _ = correr_episodio(controlador, escenario, semilla, llegadas)
                    filas.append(dict(escenario=escenario, controlador=nombre,
                                      semilla_entrenamiento=semilla_entrenamiento,
                                      semilla_transito=semilla, **m))
        print('Evaluado:', escenario, flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(RESULTADOS / 'evaluacion.csv', index=False)
    pd.DataFrame(ventanas_usadas).to_csv(RESULTADOS / 'ventanas_prueba.csv', index=False)

    # 4) Tabla resumen: media y desvío de cada métrica por escenario y controlador.
    resumen = []
    for escenario in ESCENARIOS_PRUEBA:
        for nombre in CONTROLADORES:
            sub = df[(df['escenario'] == escenario) & (df['controlador'] == nombre)]
            fila = dict(escenario=escenario, controlador=nombre, corridas=len(sub))
            for metrica in METRICAS:
                fila[metrica] = float(np.mean(sub[metrica]))
                fila[metrica + '_desvio'] = float(np.std(sub[metrica], ddof=1))
            resumen.append(fila)
    pd.DataFrame(resumen).to_csv(RESULTADOS / 'resumen.csv', index=False)

    # 5) Diferencia de cada controlador contra el fijo, semilla por semilla (comparación pareada).
    #    Primero se promedian los 5 agentes de una misma semilla de tránsito, para no contar
    #    el mismo tránsito como 5 observaciones independientes. Después, bootstrap de 2000 remuestreos.
    pareadas = []
    for escenario in ESCENARIOS_PRUEBA:
        del_escenario = df[df['escenario'] == escenario]
        fijo = del_escenario[del_escenario['controlador'] == 'fijo']
        fijo = fijo.set_index('semilla_transito')['espera_media']
        for nombre in ('mayor_cola', 'q_learning', 'dqn'):
            agente = del_escenario[del_escenario['controlador'] == nombre]
            agente = agente.groupby('semilla_transito')['espera_media'].apply(np.mean)
            diferencias = (agente - fijo).to_numpy()   # negativo = espera menos que el fijo
            muestras = np.random.default_rng(42).choice(diferencias, (2000, len(diferencias)), replace=True)
            medias_bootstrap = muestras.mean(axis=1)
            pareadas.append(dict(escenario=escenario, controlador=nombre,
                                 diferencia_segundos=float(np.mean(diferencias)),
                                 ic95_inferior=float(np.percentile(medias_bootstrap, 2.5)),
                                 ic95_superior=float(np.percentile(medias_bootstrap, 97.5))))
    pd.DataFrame(pareadas).to_csv(RESULTADOS / 'diferencias_pareadas.csv', index=False)
    print('Resultados guardados en results/', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--elegir-fijo', action='store_true', help='volver a elegir el verde del fijo en validación')
    p.add_argument('--episodios', type=int, help='episodios de prueba por escenario (por defecto, 30)')
    args = p.parse_args()
    cfg = leer_json(RAIZ / 'configs' / 'experimento.json')
    RESULTADOS.mkdir(exist_ok=True)
    if args.elegir_fijo or not (RESULTADOS / 'seleccion_fijo.json').exists():
        elegir_fijo(cfg)
    n = args.episodios if args.episodios is not None else cfg['episodios_evaluacion']
    evaluar(cfg, n)


if __name__ == '__main__':
    main()
