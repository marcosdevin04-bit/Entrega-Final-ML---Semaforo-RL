"""Genera una demo HTML que se abre sin conexión, con historiales calculados en Python.

Uso:
    python -m scripts.demo                          # escenario cambiante
    python -m scripts.demo --escenario montevideo
    python -m scripts.demo --gif                    # además guarda una animación de respaldo
"""
import argparse
import json
from src.experimento import RAIZ, CONTROLADORES, cargar_controlador, correr_episodio
from src.datos import cargar_ventanas, llegadas_montevideo

TITULOS = {'fijo': 'Tiempos fijos', 'mayor_cola': 'Mayor cola', 'q_learning': 'Q-learning', 'dqn': 'DQN'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--escenario', default='cambiante',
                   choices=['equilibrado', 'desigual', 'cambiante', 'estres', 'montevideo'])
    p.add_argument('--semilla', type=int, default=800000, help='semilla del tránsito')
    p.add_argument('--semilla-entrenamiento', type=int, default=0, help='qué versión de los agentes usar')
    p.add_argument('--gif', action='store_true', help='guardar también una animación de respaldo')
    args = p.parse_args()

    llegadas, momento = None, ''
    if args.escenario == 'montevideo':
        momento, tasas = cargar_ventanas('prueba')[0]
        llegadas = llegadas_montevideo(tasas, args.semilla)

    datos = dict(escenario=args.escenario, semilla=args.semilla,
                 semilla_entrenamiento=args.semilla_entrenamiento, ventana=momento, controladores=[])
    for nombre in CONTROLADORES:
        controlador = cargar_controlador(nombre, args.semilla_entrenamiento)
        metricas, historial = correr_episodio(controlador, args.escenario, args.semilla,
                                              llegadas, registrar=True)
        datos['controladores'].append(dict(nombre=nombre, titulo=TITULOS[nombre],
                                           metricas=metricas, historial=historial))

    plantilla = (RAIZ / 'assets' / 'demo_template.html').read_text(encoding='utf-8')
    salida = RAIZ / 'results' / 'demo.html'
    salida.write_text(plantilla.replace('__DATOS__', json.dumps(datos, ensure_ascii=False)), encoding='utf-8')
    print('Abrir en cualquier navegador:', salida)
    if args.gif:
        guardar_gif(datos)


def guardar_gif(datos):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    colores = ['#89939f', '#dc9851', '#228d85', '#355e9b']
    tope = max(max(r['cola_a'], r['cola_b']) for c in datos['controladores'] for r in c['historial']) + 3

    def dibujar(t):
        for ax, c, color in zip(axes.flat, datos['controladores'], colores):
            ax.clear()
            r = c['historial'][t]
            ax.barh(['A', 'B'], [r['cola_a'], r['cola_b']], color=color)
            if r['fase_activa'] == -1:
                fase = 'Transición'
            else:
                fase = 'Verde A' if r['fase_activa'] == 0 else 'Verde B'
            ax.set(xlim=(0, tope), title=c['titulo'] + ' · ' + fase, xlabel='Vehículos esperando')
        fig.suptitle(f"Escenario {datos['escenario']} · segundo {t}")
        fig.tight_layout()

    animacion = FuncAnimation(fig, dibujar, frames=range(0, 600, 5), interval=80)
    animacion.save(RAIZ / 'results' / 'demo.gif', writer=PillowWriter(fps=12))
    plt.close(fig)


if __name__ == '__main__':
    main()
