"""Descarga oficial y selección reproducible de dos accesos al cruce de Bv. Artigas y Rivera.

No se imputan cruces ni se inventan mediciones: el archivo reducido conserva las
filas originales seleccionadas. Las llegadas por segundo son una simulación.

Uso:
    python -m scripts.preparar_datos                   # descarga el ZIP (unos 120 MB)
    python -m scripts.preparar_datos --zip ruta.zip    # usa un ZIP ya descargado
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / 'data'
URL = ('https://ckan-data.montevideo.gub.uy/dataset/'
       'a2b2243b-358f-4f03-b7bc-0a32c25ed10c/resource/'
       'c839b3cd-d12f-4308-976e-d4e9ff9d4d3e/download/autoscope_01_2026_volumen.zip')
# (avenida, desde, hacia) -> acceso del proyecto
UBICACIONES = {
    ('Bv Artigas', 'Palmar', 'Rivera'): 'A',
    ('Rivera', 'Ponce', 'Bv Artigas'): 'B',
}
DETECTORES_A = {'701411', '701412', '701413'}
DETECTORES_B = {'701404', '701405'}


def particion_de(momento):
    """Separación por fechas de enero: 1-20 entrenamiento, 21-25 validación, 26-31 prueba."""
    if momento.day <= 20:
        return 'entrenamiento'
    if momento.day <= 25:
        return 'validacion'
    return 'prueba'


def preparar(ruta_zip):
    DATOS.mkdir(exist_ok=True)
    seleccionadas, agrupadas = [], defaultdict(list)
    filas_leidas = 0

    # 1) Leer el CSV del ZIP fila por fila (son 8,4 millones) y quedarse con los dos accesos.
    with zipfile.ZipFile(ruta_zip) as z:
        miembro = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with io.TextIOWrapper(z.open(miembro), encoding='utf-8-sig') as f:
            for fila in csv.DictReader(f):
                filas_leidas += 1
                ubicacion = tuple(fila[k] for k in ('dsc_avenida', 'dsc_int_anterior', 'dsc_int_siguiente'))
                if ubicacion not in UBICACIONES:
                    continue
                acceso = UBICACIONES[ubicacion]
                volumen = float(fila['volumen_hora'])
                if volumen < 0:
                    continue
                seleccionadas.append(fila)
                # Si hubiera duplicados del mismo detector e instante, se promedian antes de sumar carriles.
                clave = (fila['fecha'], fila['hora'], acceso, fila['cod_detector'])
                agrupadas[clave].append(volumen)
    if not seleccionadas:
        raise ValueError('No se encontraron las ubicaciones seleccionadas')
    with (DATOS / 'montevideo_seleccion.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(seleccionadas[0]))
        w.writeheader()
        w.writerows(seleccionadas)

    # 2) Por cada instante, guardar el volumen de cada detector (de 06:00 a 21:55).
    por_momento = defaultdict(dict)
    for (fecha, hora, acceso, detector), valores in agrupadas.items():
        momento = datetime.strptime(fecha + ' ' + hora, '%Y-%m-%d %H:%M:%S')
        if momento.hour < 6 or momento.hour >= 22:
            continue
        por_momento[momento].setdefault(acceso, {})[detector] = sum(valores) / len(valores)

    # 3) Sumar carriles por acceso, solo si están los cinco detectores, y pasar a autos por segundo.
    perfiles = []
    for momento, accesos in sorted(por_momento.items()):
        if (set(accesos) != {'A', 'B'} or set(accesos['A']) != DETECTORES_A
                or set(accesos['B']) != DETECTORES_B):
            continue
        # Se toma volumen_hora como veh/h, según el catálogo. Ver scripts/sensibilidad_montevideo.py:
        # hay evidencia de que en este archivo el valor es el conteo de los 5 minutos.
        perfiles.append(dict(momento=momento.isoformat(' '), particion=particion_de(momento),
                             tasa_a=sum(accesos['A'].values()) / 3600,
                             tasa_b=sum(accesos['B'].values()) / 3600))
    with (DATOS / 'perfiles_montevideo.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['momento', 'particion', 'tasa_a', 'tasa_b'])
        w.writeheader()
        w.writerows(perfiles)

    # 4) Auditoría: de dónde salió todo y cuántas filas quedaron.
    auditoria = dict(
        url_fuente=URL, fecha_descarga='2026-11-12',
        sha256_zip=hashlib.sha256(Path(ruta_zip).read_bytes()).hexdigest(),
        archivo_en_zip=miembro, filas_originales=filas_leidas,
        filas_seleccionadas=len(seleccionadas), perfiles=len(perfiles),
        duplicados_detector_momento=sum(len(v) - 1 for v in agrupadas.values()),
        perfiles_por_particion={p: sum(r['particion'] == p for r in perfiles)
                                for p in ('entrenamiento', 'validacion', 'prueba')},
        ubicaciones=[list(k) + [v] for k, v in UBICACIONES.items()],
        detectores=sorted(set(r['cod_detector'] for r in seleccionadas)),
        nota='Tasas derivadas de volumen_hora. No son llegadas individuales observadas.')
    (DATOS / 'auditoria_datos.json').write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(auditoria, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--zip', type=Path, help='ZIP ya descargado; evita usar internet')
    args = p.parse_args()
    ruta = args.zip or DATOS / 'autoscope_01_2026_volumen.zip'
    if not ruta.exists():
        DATOS.mkdir(exist_ok=True)
        print('Descargando ZIP oficial de aproximadamente 120 MB...', flush=True)
        urllib.request.urlretrieve(URL, ruta)
    preparar(ruta)


if __name__ == '__main__':
    main()
