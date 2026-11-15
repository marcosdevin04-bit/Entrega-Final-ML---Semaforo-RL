"""Descarga oficial y selección reproducible de dos aproximaciones al cruce.

No se imputan cruces ni se inventan mediciones: el archivo reducido conserva las
filas originales seleccionadas. Las llegadas por segundo son una simulación.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
URL = ('https://ckan-data.montevideo.gub.uy/dataset/'
       'a2b2243b-358f-4f03-b7bc-0a32c25ed10c/resource/'
       'c839b3cd-d12f-4308-976e-d4e9ff9d4d3e/download/autoscope_01_2026_volumen.zip')
LOCATIONS = {
    ('Bv Artigas', 'Palmar', 'Rivera'): 'A',
    ('Rivera', 'Ponce', 'Bv Artigas'): 'B',
}

def prepare(zip_path):
    DATA.mkdir(exist_ok=True)
    selected, grouped = [], defaultdict(list)
    rows_read = 0
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with io.TextIOWrapper(z.open(member), encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                rows_read += 1
                location = tuple(row[k] for k in ('dsc_avenida', 'dsc_int_anterior', 'dsc_int_siguiente'))
                if location not in LOCATIONS:
                    continue
                direction = LOCATIONS[location]
                volume = float(row['volumen_hora'])
                if volume < 0:
                    continue
                selected.append(row)
                # Agrupar duplicados del mismo detector e instante antes de sumar carriles.
                key = (row['fecha'], row['hora'], direction, row['cod_detector'])
                grouped[key].append(volume)
    if not selected:
        raise ValueError('No se encontraron las ubicaciones seleccionadas')
    with (DATA / 'montevideo_seleccion.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)
    directions = defaultdict(dict)
    for (date, time, direction, detector), values in grouped.items():
        stamp = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
        if stamp.hour < 6 or stamp.hour >= 22:
            continue
        directions[stamp].setdefault(direction, {})[detector] = sum(values) / len(values)
    profiles = []
    for stamp, sides in sorted(directions.items()):
        if (set(sides) != {'A', 'B'} or
                set(sides['A']) != {'701411', '701412', '701413'} or
                set(sides['B']) != {'701404', '701405'}):
            continue
        split = 'train' if stamp.day <= 20 else ('validation' if stamp.day <= 25 else 'test')
        # Conteos expresados en veh/h -> intensidad de Poisson en veh/s.
        profiles.append(dict(timestamp=stamp.isoformat(' '), split=split,
                             rate_a=sum(sides['A'].values()) / 3600,
                             rate_b=sum(sides['B'].values()) / 3600))
    with (DATA / 'perfiles_montevideo.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'split', 'rate_a', 'rate_b'])
        w.writeheader()
        w.writerows(profiles)
    audit = dict(source_url=URL, downloaded_local_date='2026-11-12',
                 zip_sha256=hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
                 archive_member=member, rows_original=rows_read,
                 rows_selected=len(selected), profiles=len(profiles),
                 duplicate_detector_timestamps=sum(len(v)-1 for v in grouped.values()),
                 counts_by_split={s:sum(r['split']==s for r in profiles) for s in ('train','validation','test')},
                 locations=[list(k) + [v] for k,v in LOCATIONS.items()],
                 detectors=sorted(set(r['cod_detector'] for r in selected)),
                 note='Tasas derivadas de volumen_hora. No son llegadas individuales observadas.')
    (DATA / 'auditoria_datos.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, help='ZIP ya descargado; evita usar internet')
    args = p.parse_args()
    path = args.zip or DATA / 'autoscope_01_2026_volumen.zip'
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print('Descargando ZIP oficial de aproximadamente 120 MB...', flush=True)
        urllib.request.urlretrieve(URL, path)
    prepare(path)

if __name__ == '__main__':
    main()
