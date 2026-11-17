"""Análisis de sensibilidad: ¿qué mide realmente `volumen_hora`?

El catálogo describe `volumen_hora` como autos por hora (conteo de 5 minutos x 12).
Este script junta la evidencia de que, en el archivo de enero de 2026, la columna
trae en realidad el conteo de los 5 minutos, y vuelve a evaluar los cuatro
controladores en Montevideo con esa lectura (tasas x 12).

No se reentrena ni se ajusta nada: se usan los mismos modelos, las mismas ventanas
de prueba y las mismas semillas que en `scripts/evaluar.py`.

Uso:
    python -m scripts.sensibilidad_montevideo
"""
import json
import numpy as np
import pandas as pd
from src.datos import cargar_ventanas, llegadas_montevideo
from src.experimento import RAIZ, CONTROLADORES, leer_json, cargar_controlador, correr_episodio

RESULTADOS = RAIZ / 'results'
FACTOR = 12   # una hora tiene 12 bloques de 5 minutos

# Indicador INEX0105 del Observatorio de Movilidad (IM): vehículos por mes en
# Av. Rivera, tramo Bv. Gral. Artigas -> Av. Ing. L. P. Ponce (sentido B), 2017-2018.
# Es la misma cuadra que el acceso B del proyecto, en el sentido contrario.
FUENTE_IM = ('https://montevideo.gub.uy/sites/default/files/biblioteca/'
             'cantidaddevehiculosencirculacionporejevialseleccionado_3.pdf')
RIVERA_IM_MIN, RIVERA_IM_MAX = 295_197, 401_215
DETECTORES = {'701411': 'A', '701412': 'A', '701413': 'A', '701404': 'B', '701405': 'B'}


def evidencia_unidad():
    """Dos comprobaciones sobre las filas originales de enero de 2026."""
    df = pd.read_csv(RAIZ / 'data' / 'montevideo_seleccion.csv', dtype={'cod_detector': str})
    valores = df['volumen_hora']

    # 1) Si fuera (conteo de 5 minutos) x 12, todos los valores serían múltiplos de 12.
    positivos = valores[valores > 0]
    pct_multiplos = float((positivos % 12 == 0).mean() * 100)
    mas_frecuentes = positivos.value_counts().head(5).index.astype(int).tolist()

    # 2) Total del mes por acceso con cada lectura.
    #    Lectura "por hora": en 5 minutos pasan valor / 12 autos.
    #    Lectura "5 minutos": en 5 minutos pasan valor autos.
    df['acceso'] = df['cod_detector'].map(DETECTORES)
    suma = df.groupby('acceso')['volumen_hora'].sum()
    totales = {acceso: dict(lectura_por_hora=round(float(s) / FACTOR),
                            lectura_5_minutos=round(float(s)))
               for acceso, s in suma.items()}

    evidencia = dict(
        registros=int(len(valores)),
        registros_positivos=int(len(positivos)),
        porcentaje_multiplos_de_12=round(pct_multiplos, 1),
        valores_mas_frecuentes=mas_frecuentes,
        total_enero_por_acceso=totales,
        referencia_im_rivera_vehiculos_por_mes=[RIVERA_IM_MIN, RIVERA_IM_MAX],
        fuente_referencia=FUENTE_IM)
    (RESULTADOS / 'unidad_volumen_hora.json').write_text(
        json.dumps(evidencia, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Valores > 0 que son múltiplos de 12: {pct_multiplos:.1f} %')
    print(f'Valores más frecuentes: {mas_frecuentes}')
    print(f"Rivera (acceso B) en enero: {totales['B']['lectura_por_hora']:,} por hora "
          f"vs {totales['B']['lectura_5_minutos']:,} por 5 minutos; "
          f'IM informa {RIVERA_IM_MIN:,} a {RIVERA_IM_MAX:,} por mes')


def evaluar_con_factor(cfg):
    """Misma prueba de Montevideo que evaluar.py, con las tasas multiplicadas por 12."""
    ventanas = cargar_ventanas('prueba')
    n = cfg['episodios_evaluacion']
    rng = np.random.default_rng(2026)          # mismas ventanas que evaluar.py
    idx = rng.choice(len(ventanas), size=n, replace=n > len(ventanas))

    filas = []
    for nombre in CONTROLADORES:
        semillas = cfg['semillas_entrenamiento'] if nombre in ('q_learning', 'dqn') else [-1]
        for semilla_entrenamiento in semillas:
            controlador = cargar_controlador(nombre, semilla_entrenamiento)
            for i in range(n):
                semilla = cfg['semilla_evaluacion_inicial'] + i   # mismas semillas que evaluar.py
                _, tasas = ventanas[idx[i]]
                for factor in (1, FACTOR):
                    llegadas = llegadas_montevideo(tasas * factor, semilla)
                    m, _ = correr_episodio(controlador, 'montevideo', semilla, llegadas)
                    filas.append(dict(factor=factor, controlador=nombre,
                                      semilla_entrenamiento=semilla_entrenamiento,
                                      semilla_transito=semilla, **m))
    df = pd.DataFrame(filas)

    # Tabla resumen, igual que resumen.csv.
    resumen = []
    for factor in (1, FACTOR):
        for nombre in CONTROLADORES:
            sub = df[(df['factor'] == factor) & (df['controlador'] == nombre)]
            fila = dict(factor=factor, controlador=nombre, corridas=len(sub))
            for metrica in ('espera_media', 'espera_p95', 'pendientes', 'cambios'):
                fila[metrica] = float(np.mean(sub[metrica]))
                fila[metrica + '_desvio'] = float(np.std(sub[metrica], ddof=1))
            resumen.append(fila)
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(RESULTADOS / 'sensibilidad_montevideo.csv', index=False)

    # Diferencias pareadas, semilla por semilla, con bootstrap (igual que evaluar.py).
    # Primero se promedian las 5 versiones de cada agente en la misma semilla de tránsito.
    # Se compara cada controlador contra el fijo y, además, DQN contra mayor cola.
    comparaciones = [('mayor_cola', 'fijo'), ('q_learning', 'fijo'), ('dqn', 'fijo'), ('dqn', 'mayor_cola')]
    pareadas = []
    for factor in (1, FACTOR):
        del_factor = df[df['factor'] == factor]
        por_semilla = del_factor.groupby(['controlador', 'semilla_transito'])['espera_media'].apply(np.mean)
        for nombre, referencia in comparaciones:
            diferencias = (por_semilla[nombre] - por_semilla[referencia]).to_numpy()  # negativo = espera menos
            medias_bootstrap = np.random.default_rng(42).choice(diferencias, (2000, len(diferencias))).mean(axis=1)
            pareadas.append(dict(factor=factor, controlador=nombre, referencia=referencia,
                                 diferencia_segundos=float(np.mean(diferencias)),
                                 ic95_inferior=float(np.percentile(medias_bootstrap, 2.5)),
                                 ic95_superior=float(np.percentile(medias_bootstrap, 97.5))))
    pd.DataFrame(pareadas).to_csv(RESULTADOS / 'sensibilidad_montevideo_pareadas.csv', index=False)

    tabla = resumen.pivot(index='factor', columns='controlador', values='espera_media')
    print('\nEspera media por auto en Montevideo (s):')
    print(tabla[list(CONTROLADORES)].round(2).to_string())


def main():
    cfg = leer_json(RAIZ / 'configs' / 'experimento.json')
    RESULTADOS.mkdir(exist_ok=True)
    evidencia_unidad()
    evaluar_con_factor(cfg)
    print('\nResultados guardados en results/')


if __name__ == '__main__':
    main()
