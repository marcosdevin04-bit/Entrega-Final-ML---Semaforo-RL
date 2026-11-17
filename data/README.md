# Origen y preparación de los datos

## Nota de origen

**Proveedor:** Intendencia de Montevideo, Departamento de Movilidad, Centro de
Gestión de Movilidad (CGM).

**Conjunto:** Conteo vehicular en las principales avenidas de Montevideo.

**Recurso utilizado:** Conteo Enero 2026, `autoscope_01_2026_volumen.zip`.

**Consulta y descarga:** 12 de noviembre de 2026.

**Licencia:** Licencia de Datos Abiertos - Uruguay, versión 0.1, indicada por el
catálogo oficial. Esta nota identifica al proveedor, conjunto y modificaciones.

- [Página oficial del conjunto](https://catalogodatos.gub.uy/dataset/conteo-vehicular-en-las-principales-avenidas-de-montevideo)
- [Descarga del ZIP utilizado](https://ckan-data.montevideo.gub.uy/dataset/a2b2243b-358f-4f03-b7bc-0a32c25ed10c/resource/c839b3cd-d12f-4308-976e-d4e9ff9d4d3e/download/autoscope_01_2026_volumen.zip)
- [Texto de la licencia](https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/documentos/publicaciones/licencia_de_datos_abiertos_0.pdf)

No se atribuye a la Intendencia la simulación, los modelos ni las conclusiones.

## Archivos incluidos

| Archivo | Contenido |
|---|---|
| `montevideo_seleccion.csv` | 44.578 filas originales de los dos accesos elegidos, sin cambiar sus valores. |
| `perfiles_montevideo.csv` | 5.941 registros derivados con tasas para ambos accesos. |
| `auditoria_datos.json` | URL, hash SHA-256 del ZIP, cantidad de filas y detectores. |
| `metadata_conteo_original.txt` | Metadatos publicados por el proveedor, sin modificaciones. |
| `catalogo_original.json` | Respuesta del catálogo guardada al consultar la fuente. |

El ZIP original tiene aproximadamente 120 MB y contiene un CSV de unos 760 MB,
con 8.425.612 filas. Se incluye una selección pequeña para que GitHub y la ejecución
local resulten manejables. El ZIP se puede volver a descargar mediante:

```powershell
python -m scripts.preparar_datos
```

También se puede usar un archivo descargado:

```powershell
python -m scripts.preparar_datos --zip ruta/al/autoscope_01_2026_volumen.zip
```

## Selección de accesos

| Acceso del simulador | Avenida | Cruce anterior | Cruce siguiente | Detectores |
|---|---|---|---|---|
| A | Bv Artigas | Palmar | Rivera | 701411, 701412, 701413 |
| B | Rivera | Ponce | Bv Artigas | 701404, 701405 |

Se usan dos aproximaciones al entorno de Bulevar Artigas y Rivera. No se reconstruye
la geometría completa ni se modelan todos los movimientos reales. Las etiquetas
norte-sur y este-oeste del código son nombres simplificados de A y B.

## Columnas originales

- `cod_detector`: identificador del detector.
- `id_carril`: identificador del carril.
- `fecha`, `hora`: momento del registro, tratado en hora local tal como lo publica la fuente.
- `dsc_avenida`: vía donde se realiza la medición.
- `dsc_int_anterior`, `dsc_int_siguiente`: cruces que describen el sentido medido.
- `latitud`, `longitud`: coordenadas publicadas del sensor.
- `volumen_hora`: volumen expresado como veh/h según el nombre del campo y el catálogo.

Hay una diferencia entre la descripción general y este recurso: la página
menciona `volumen` y `volumen_hora`, pero el CSV elegido contiene únicamente
`volumen_hora`. Además, el TXT habla de la última hora y la página describe una
estimación horaria a partir de cinco minutos. **No se reconstruyen conteos exactos
de cinco minutos ni llegadas individuales** a partir de esa discrepancia.

## Transformaciones

1. Filtrar por los nombres exactos de las dos ubicaciones anteriores.
2. Leer la hora con un formato que admite segundos como `0`, presentes en el CSV.
3. Excluir volúmenes negativos. No se reemplazan faltantes por cero.
4. Si existieran duplicados por detector y timestamp, promediar su volumen antes
   de sumar carriles. En la selección utilizada no se detectaron duplicados.
5. Conservar únicamente timestamps con los cinco detectores esperados presentes.
6. Conservar horarios de 06:00 a 21:55, inclusive. Es una elección de alcance del
   proyecto, no una recomendación del proveedor.
7. Sumar `volumen_hora` de los carriles de cada acceso y dividir entre 3.600 para
   obtener una tasa en vehículos por segundo.
8. Separar fechas: días 1-20, entrenamiento; 21-25, validación; 26-31, pruebas.
9. Formar ventanas de dos registros consecutivos separados por cinco minutos.
   Como cada registro describe los cinco minutos previos, cada ventana proporciona
   dos bloques de 300 segundos sin comprimir el tiempo.
10. Generar llegadas de Poisson por segundo usando esas tasas. Estas llegadas son
    sintéticas y se comparten exactamente entre los cuatro controladores.

Los perfiles contienen 3.835 registros de entrenamiento, 954 de validación y
1.152 de prueba. Los episodios usan ventanas completas dentro de cada partición.

## Límites de interpretación

Los sensores observan vehículos que circulan por el punto de medición, no la
demanda latente que habría llegado bajo otra política semafórica. Usar esas
mediciones como intensidad exógena es una aproximación. Los datos tampoco traen
las acciones del semáforo, colas o esperas para validar las salidas del simulador.

Las tasas obtenidas bajo la unidad publicada son bajas para este modelo. El
resultado local debe interpretarse condicionado a esa conversión, al período de
enero y a estos accesos.

Hay evidencia de que, en este archivo, `volumen_hora` contiene el conteo de los
cinco minutos y no veh/h: solo el 6,5 % de los valores positivos son múltiplos de
12 (deberían serlo todos si fuera conteo x 12), y el total de enero del acceso B
solo coincide con el indicador mensual de la Intendencia para esa cuadra si se lee
como conteo de cinco minutos. `scripts/sensibilidad_montevideo.py` reproduce estas
comprobaciones y evalúa los controladores con esa lectura. Para una recomendación
vial haría falta confirmar con el proveedor la semántica del campo, la calidad de
cada sensor y las capacidades de descarga de la intersección.
