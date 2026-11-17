"""Exporta episodios del simulador del proyecto para la visualización 3D.

Corre cada controlador en varios escenarios con el código del repo (src/) y guarda,
para cada auto, en qué segundo llegó y en qué segundo cruzó. Así la animación
muestra exactamente lo que calculó Python, no un tránsito inventado.

Uso (desde la raíz del repo, con el entorno activado):
    python extra/simulador_3d/exportar_datos.py
"""
import argparse
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def exportar(repo):
    sys.path.insert(0, str(repo))
    import numpy as np
    from src.entorno import Cruce
    from src.datos import cargar_ventanas, llegadas_montevideo
    from src.experimento import CONTROLADORES, NOMBRES, cargar_controlador

    semilla = 800000   # primera semilla de la prueba oficial
    # Misma primera ventana de prueba que usa la evaluación oficial (rng 2026).
    ventanas = cargar_ventanas('prueba')
    idx = np.random.default_rng(2026).choice(len(ventanas), size=30, replace=False)
    momento, tasas = ventanas[idx[0]]

    escenarios = [
        ('cambiante', 'Demanda cambiante', 'A cargada la primera mitad, B la segunda.', None),
        ('desigual', 'Demanda desigual', 'Bv. Artigas con 4 veces más tránsito que Rivera.', None),
        ('equilibrado', 'Demanda equilibrada', 'Las dos calles con el mismo tránsito.', None),
        ('estres', 'Estrés', 'Mucho tránsito en las dos calles. Los agentes no lo vieron al entrenar.', None),
        ('montevideo_x12', 'Montevideo (conteo de 5 min)',
         f'Sensores del {momento[:10]} a las {momento[11:16]}, leyendo volumen_hora como conteo de 5 minutos.',
         llegadas_montevideo(tasas * 12, semilla)),
        ('montevideo', 'Montevideo (unidad publicada)',
         f'Los mismos sensores leídos como veh/h: el cruce queda casi vacío.',
         llegadas_montevideo(tasas, semilla)),
    ]

    salida = dict(semilla_transito=semilla, semilla_entrenamiento=0, duracion=600,
                  nombres=NOMBRES, escenarios=[])
    for clave, titulo, descripcion, llegadas in escenarios:
        esc = dict(id=clave, titulo=titulo, descripcion=descripcion, controladores={})
        for nombre in CONTROLADORES:
            controlador = cargar_controlador(nombre, 0)
            cruce = Cruce()
            obs = cruce.reiniciar(clave if llegadas is None else 'montevideo', semilla, llegadas)
            termino = False
            while not termino:
                obs, _, termino = cruce.avanzar(controlador.actuar(obs), registrar=True)
            h = cruce.historial
            lleg = cruce.llegadas

            # Reconstruir cada auto: las colas son FIFO, así que alcanza con saber
            # cuántos llegaron y cuántos salieron de cada calle en cada segundo.
            autos = []
            for calle, columna in ((0, 'cola_a'), (1, 'cola_b')):
                esperando = []          # índices de autos en cola, en orden de llegada
                cola_anterior = 0
                for t in range(len(h)):
                    for _ in range(int(lleg[t, calle])):
                        autos.append([calle, t, -1])
                        esperando.append(len(autos) - 1)
                    salieron = cola_anterior + int(lleg[t, calle]) - h[t][columna]
                    for _ in range(salieron):
                        autos[esperando.pop(0)][2] = t
                    cola_anterior = h[t][columna]
            autos.sort(key=lambda a: (a[1], a[0]))

            m = cruce.metricas()
            esc['controladores'][nombre] = dict(
                fases=''.join('t' if r['fase_activa'] == -1 else 'ab'[r['fase_activa']] for r in h),
                cola_a=[r['cola_a'] for r in h], cola_b=[r['cola_b'] for r in h],
                espera=[r['espera_total'] for r in h], cruzaron=[r['cruzaron'] for r in h],
                autos=autos,
                final=dict(espera_media=round(m['espera_media'], 2), espera_p95=round(m['espera_p95'], 1),
                           cruzaron=m['cruzaron'], pendientes=m['pendientes'], cambios=m['cambios']))
        salida['escenarios'].append(esc)
        print('Exportado:', titulo)

    texto = 'window.DATOS = ' + json.dumps(salida, ensure_ascii=False, separators=(',', ':')) + ';\n'
    (AQUI / 'datos.js').write_text(texto, encoding='utf-8')
    print(f'datos.js: {len(texto) // 1024} KB')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    exportar(AQUI.parents[1])   # la raíz del repo está dos carpetas más arriba
