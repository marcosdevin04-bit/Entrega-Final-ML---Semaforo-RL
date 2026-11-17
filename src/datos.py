"""Perfiles de tránsito de Montevideo. Ningún entrenamiento necesita internet."""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PERFILES = Path(__file__).resolve().parents[1] / 'data' / 'perfiles_montevideo.csv'


def cargar_ventanas(particion):
    """Ventanas de dos registros consecutivos (5 + 5 minutos) de una partición.

    particion: 'entrenamiento' (1 al 20 de enero), 'validacion' (21 al 25) o 'prueba' (26 al 31).
    Devuelve una lista de (momento, tasas), donde tasas es una matriz 2 x 2:
    una fila por registro y una columna por calle, en autos por segundo.
    """
    perfiles = pd.read_csv(PERFILES)
    filas = perfiles[perfiles['particion'] == particion].to_dict('records')
    ventanas = []
    for a, b in zip(filas, filas[1:]):
        separacion = datetime.fromisoformat(b['momento']) - datetime.fromisoformat(a['momento'])
        if separacion.total_seconds() == 300:   # solo registros seguidos
            tasas = np.array([[a['tasa_a'], a['tasa_b']],
                              [b['tasa_a'], b['tasa_b']]], dtype=float)
            ventanas.append((a['momento'], tasas))
    if not ventanas:
        raise ValueError('No hay ventanas consecutivas para ' + particion)
    return ventanas


def llegadas_montevideo(tasas, semilla):
    """Autos por segundo para un episodio de 600 s a partir de una ventana.

    Cada registro representa los 5 minutos previos a su momento: se repite su tasa
    durante 300 segundos, sin comprimir el tiempo, y se sortean llegadas de Poisson.
    """
    tasas_por_segundo = np.repeat(tasas, 300, axis=0)
    return np.random.default_rng(semilla).poisson(tasas_por_segundo)
