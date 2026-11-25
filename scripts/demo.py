"""Abre el simulador 3D del cruce con los episodios calculados por el proyecto.

Corre los cuatro controladores en seis escenarios con el código de src/ y guarda,
para cada auto, en qué segundo llegó y en qué segundo cruzó (demo/datos.js). Después
levanta un servidor local y abre demo/index.html en el navegador. La animación
muestra exactamente lo que calculó Python: no es tránsito inventado.

Uso:
    python -m scripts.demo                # genera los datos y abre el simulador
    python -m scripts.demo --solo-datos   # solo regenera demo/datos.js

Para cerrar el servidor: Ctrl+C. Necesita internet para cargar Three.js.
"""
import argparse
import functools
import http.server
import json
import webbrowser
import numpy as np
from src.entorno import Cruce
from src.datos import cargar_ventanas, llegadas_montevideo
from src.experimento import RAIZ, CONTROLADORES, NOMBRES, cargar_controlador

CARPETA = RAIZ / 'demo'
SEMILLA = 800000   # primera semilla de tránsito de la prueba oficial


def escenarios():
    """Los cuatro sintéticos y Montevideo con las dos lecturas de volumen_hora."""
    # Misma primera ventana de prueba que usa la evaluación oficial (generador 2026).
    ventanas = cargar_ventanas('prueba')
    idx = np.random.default_rng(2026).choice(len(ventanas), size=30, replace=False)
    momento, tasas = ventanas[idx[0]]
    return [
        ('cambiante', 'Demanda cambiante', 'A cargada la primera mitad, B la segunda.', None),
        ('desigual', 'Demanda desigual', 'Bv. Artigas con 4 veces más tránsito que Rivera.', None),
        ('equilibrado', 'Demanda equilibrada', 'Las dos calles con el mismo tránsito.', None),
        ('estres', 'Estrés', 'Mucho tránsito en las dos calles. Los agentes no lo vieron al entrenar.', None),
        ('montevideo_x12', 'Montevideo (conteo de 5 min)',
         f'Sensores del {momento[:10]} a las {momento[11:16]}, leyendo volumen_hora como conteo de 5 minutos.',
         llegadas_montevideo(tasas * 12, SEMILLA)),
        ('montevideo', 'Montevideo (unidad publicada)',
         'Los mismos sensores leídos como veh/h: el cruce queda casi vacío.',
         llegadas_montevideo(tasas, SEMILLA)),
    ]


def reconstruir_autos(historial, llegadas):
    """Llegada y cruce de cada auto. Las colas son FIFO: alcanza con saber
    cuántos llegaron y cuántos salieron de cada calle en cada segundo."""
    autos = []
    for calle, columna in ((0, 'cola_a'), (1, 'cola_b')):
        esperando = []          # índices de autos en cola, en orden de llegada
        cola_anterior = 0
        for t, registro in enumerate(historial):
            for _ in range(int(llegadas[t, calle])):
                autos.append([calle, t, -1])   # -1: todavía no cruzó
                esperando.append(len(autos) - 1)
            salieron = cola_anterior + int(llegadas[t, calle]) - registro[columna]
            for _ in range(salieron):
                autos[esperando.pop(0)][2] = t
            cola_anterior = registro[columna]
    autos.sort(key=lambda a: (a[1], a[0]))
    return autos


def exportar_datos():
    salida = dict(semilla_transito=SEMILLA, semilla_entrenamiento=0, duracion=600,
                  nombres=NOMBRES, escenarios=[])
    for clave, titulo, descripcion, llegadas in escenarios():
        esc = dict(id=clave, titulo=titulo, descripcion=descripcion, controladores={})
        for nombre in CONTROLADORES:
            controlador = cargar_controlador(nombre, 0)
            cruce = Cruce()
            obs = cruce.reiniciar(clave if llegadas is None else 'montevideo', SEMILLA, llegadas)
            termino = False
            while not termino:
                obs, _, termino = cruce.avanzar(controlador.actuar(obs), registrar=True)
            h = cruce.historial
            m = cruce.metricas()
            esc['controladores'][nombre] = dict(
                fases=''.join('t' if r['fase_activa'] == -1 else 'ab'[r['fase_activa']] for r in h),
                cola_a=[r['cola_a'] for r in h], cola_b=[r['cola_b'] for r in h],
                espera=[r['espera_total'] for r in h], cruzaron=[r['cruzaron'] for r in h],
                autos=reconstruir_autos(h, cruce.llegadas),
                final=dict(espera_media=round(m['espera_media'], 2), espera_p95=round(m['espera_p95'], 1),
                           cruzaron=m['cruzaron'], pendientes=m['pendientes'], cambios=m['cambios']))
        salida['escenarios'].append(esc)
        print('Episodio listo:', titulo, flush=True)
    texto = 'window.DATOS = ' + json.dumps(salida, ensure_ascii=False, separators=(',', ':')) + ';\n'
    (CARPETA / 'datos.js').write_text(texto, encoding='utf-8')


def abrir_simulador(puerto):
    """Sirve la carpeta demo/ en este equipo y abre el navegador."""
    class Silencioso(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    manejador = functools.partial(Silencioso, directory=str(CARPETA))
    try:
        servidor = http.server.ThreadingHTTPServer(('127.0.0.1', puerto), manejador)
    except OSError:   # el puerto está ocupado: se usa uno libre
        servidor = http.server.ThreadingHTTPServer(('127.0.0.1', 0), manejador)
    url = f'http://127.0.0.1:{servidor.server_address[1]}/index.html'
    print(f'\nSimulador 3D en {url}\nPara cerrarlo: Ctrl+C', flush=True)
    webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print('\nServidor cerrado.')
    finally:
        servidor.server_close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--solo-datos', action='store_true', help='regenerar demo/datos.js sin abrir el navegador')
    p.add_argument('--puerto', type=int, default=8000)
    args = p.parse_args()
    exportar_datos()
    if not args.solo_datos:
        abrir_simulador(args.puerto)


if __name__ == '__main__':
    main()
