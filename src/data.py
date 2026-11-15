"""Perfiles locales de Montevideo. Ningún entrenamiento necesita internet."""
from pathlib import Path
import csv
from datetime import datetime
import numpy as np

def load_windows(split):
    path = Path(__file__).resolve().parents[1] / 'data' / 'perfiles_montevideo.csv'
    with path.open(encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == split]
    windows = []
    for a, b in zip(rows, rows[1:]):
        dt = datetime.fromisoformat(b['timestamp']) - datetime.fromisoformat(a['timestamp'])
        if dt.total_seconds() == 300:
            windows.append((a['timestamp'], np.array([[float(a['rate_a']), float(a['rate_b'])],
                                                      [float(b['rate_a']), float(b['rate_b'])]])))
    if not windows:
        raise ValueError('No hay ventanas consecutivas para ' + split)
    return windows

def real_arrivals(window, seed, horizon=600):
    # Cada fila representa los 5 minutos previos al timestamp publicado.
    # Los dos bloques se reproducen sin comprimir el tiempo (episodio de 600 s).
    if horizon != 600:
        raise ValueError('Los perfiles de dos registros requieren episodios de 600 segundos')
    rates = np.repeat(window, 300, axis=0)
    return np.random.default_rng(seed).poisson(rates)
