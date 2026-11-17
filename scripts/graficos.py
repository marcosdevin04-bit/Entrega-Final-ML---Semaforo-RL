"""Genera los gráficos de results/figures/ a partir de los CSV de resultados.

Uso:
    python -m scripts.graficos
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # guarda imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from src.experimento import RAIZ, CONTROLADORES, NOMBRES, cargar_controlador, correr_episodio

COLORES = ['#89939f', '#dc9851', '#228d85', '#355e9b']
ESCENARIOS = ['equilibrado', 'desigual', 'cambiante', 'estres', 'montevideo']
ETIQUETAS = ['Equilibrado', 'Desigual', 'Cambiante', 'Estrés', 'Montevideo']


def barras_por_escenario(resumen, carpeta):
    """Un gráfico de barras por métrica: media y desvío de cada controlador en cada escenario."""
    for metrica, titulo, eje_y in [
            ('espera_media', 'Espera observada por vehículo', 'Segundos por vehículo'),
            ('espera_p95', 'Percentil 95 de espera observada', 'Segundos'),
            ('pendientes', 'Vehículos pendientes al finalizar', 'Vehículos')]:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(ESCENARIOS))
        for j, nombre in enumerate(CONTROLADORES):
            del_controlador = resumen[resumen['controlador'] == nombre].set_index('escenario').loc[ESCENARIOS]
            ax.bar(x + (j - 1.5) * .19, del_controlador[metrica], width=.18, label=NOMBRES[nombre],
                   color=COLORES[j], yerr=del_controlador[metrica + '_desvio'],
                   capsize=2, error_kw={'lw': .7})
        ax.set_xticks(x, ETIQUETAS)
        ax.set(ylabel=eje_y, title=titulo + ' (media y desvío)')
        ax.legend(ncol=4, frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(carpeta / (metrica + '.png'))
        plt.close(fig)


def curvas_de_aprendizaje(carpeta):
    """Espera durante el entrenamiento: media móvil de 40 episodios, una línea fina por semilla."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, algoritmo, color in zip(axes, ['q_learning', 'dqn'], [COLORES[2], COLORES[3]]):
        curvas = []
        for ruta in sorted((RAIZ / 'results').glob(f'entrenamiento_{algoritmo}_*.csv')):
            esperas = pd.read_csv(ruta)['espera_media'].to_numpy()
            suavizada = np.convolve(esperas, np.ones(40) / 40, mode='valid')
            ax.plot(np.arange(len(suavizada)) + 40, suavizada, color=color, alpha=.25, lw=.8)
            curvas.append(suavizada)
        if curvas:
            media = np.mean(curvas, axis=0)
            ax.plot(np.arange(len(media)) + 40, media, color=color, lw=2.2)
        ax.set(title=NOMBRES[algoritmo], xlabel='Episodio de entrenamiento')
    axes[0].set_ylabel('Espera observada (media móvil de 40 episodios)')
    fig.suptitle('Entrenamiento con exploración y mezcla de demandas')
    fig.tight_layout()
    fig.savefig(carpeta / 'aprendizaje.png')
    plt.close(fig)


def colas_en_un_episodio(carpeta):
    """Autos esperando segundo a segundo en un episodio de prueba, sin elegir el mejor agente."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for j, nombre in enumerate(CONTROLADORES):
        _, historial = correr_episodio(cargar_controlador(nombre, 0), 'cambiante', 800000, registrar=True)
        ax.plot([r['t'] for r in historial], [r['cola_a'] + r['cola_b'] for r in historial],
                label=NOMBRES[nombre], color=COLORES[j], alpha=.9)
    ax.axvline(300, color='black', ls='--', lw=.8, label='Cambio de demanda')
    ax.set(xlabel='Segundo', ylabel='Vehículos esperando',
           title='Mismo tránsito y fase inicial para los cuatro controles')
    ax.legend(ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(carpeta / 'colas_episodio.png')
    plt.close(fig)


def main():
    carpeta = RAIZ / 'results' / 'figures'
    carpeta.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    resumen = pd.read_csv(RAIZ / 'results' / 'resumen.csv')
    barras_por_escenario(resumen, carpeta)
    curvas_de_aprendizaje(carpeta)
    colas_en_un_episodio(carpeta)
    print('Gráficos guardados en results/figures/')


if __name__ == '__main__':
    main()
