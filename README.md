# Semáforo RL

Control adaptativo de una intersección mediante aprendizaje por refuerzo.
Proyecto final grupal de Machine Learning.

**Integrantes:** Marcos Devincenzi, Facundo Díaz, Santiago Alba y Domingo Leiva.
**Docente:** Gustavo Vazquez. **Entrega:** 26 de noviembre de 2026.

## Objetivo

Comparar tiempos fijos, prioridad a la mayor cola, Q-learning tabular y DQN bajo
las mismas condiciones de tránsito. El caso de aplicación incluye perfiles
derivados de conteos oficiales de Montevideo y escenarios sintéticos controlados.

## Motivación y antecedentes

Elegimos estudiar aprendizaje por refuerzo y aplicarlo a un problema cotidiano de movilidad. La investigación inicial identificó antecedentes municipales de control adaptativo. Nuestro trabajo aporta una simulación y una comparación propia; sus resultados no evalúan al sistema de la Intendencia.

El aprendizaje por refuerzo es un tema nuevo para el grupo: el proyecto implica
estudiar sus fundamentos, implementarlos y explicar su funcionamiento.

La documentación y las fuentes están en la sección «Motivación y antecedentes en Montevideo» del [informe](report/informe.md).

## Instalación

Usar Python 3.12 (las versiones de `requirements.txt` se probaron con esa versión). Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En macOS/Linux, activar con `source .venv/bin/activate`.
Los modelos incluidos se ejecutan en CPU. No se necesita GPU, cuenta ni conexión
a internet una vez instaladas las dependencias.

## Ejecutar

```powershell
# Revisar las reglas del simulador y las actualizaciones de los agentes
python -m unittest discover -s tests -v

# Comparar los modelos incluidos y generar gráficos
python -m scripts.evaluar
python -m scripts.graficos

# Revisar la unidad de los datos de Montevideo y evaluar con la lectura corregida
python -m scripts.sensibilidad_montevideo

# Abrir el simulador 3D con los episodios de los cuatro controladores
python -m scripts.demo
```

`scripts.demo` corre los cuatro controladores en seis escenarios (los sintéticos y
Montevideo con las dos lecturas de `volumen_hora`), guarda el resultado en
`demo/datos.js`, levanta un servidor local y abre `demo/index.html` en el navegador.
Para cerrarlo, `Ctrl+C` en la terminal. Necesita internet para cargar Three.js.
Los controles están en `demo/LEEME.md`.

Para entrenar nuevamente los diez agentes y reproducir el experimento completo:

```powershell
python -m scripts.entrenar
python -m scripts.evaluar --elegir-fijo
python -m scripts.graficos
python -m scripts.demo
```

La configuración está en `configs/experimento.json`. La corrida completa usa
700 episodios de 600 segundos por agente y cinco semillas por algoritmo.
También se puede probar un entrenamiento breve con
`python -m scripts.entrenar --algoritmo q_learning --episodios 100 --semillas 0`.
**Ese comando reemplaza el modelo de la semilla 0**: hacerlo en una copia si se
quieren conservar los resultados incluidos. Los modelos guardados contienen
pesos/tablas para inferencia, no el estado necesario para reanudar entrenamiento.

Para regenerar el PDF del informe después de editar `report/informe.md`:

```powershell
python -m pip install -r requirements-informe.txt
python -m scripts.informe_pdf
```

## Datos de Montevideo

Fuente: Intendencia de Montevideo, Departamento de Movilidad, Centro de Gestión
de Movilidad. Conjunto **Conteo vehicular en las principales avenidas de
Montevideo**, recurso de enero de 2026, consultado el 12 de noviembre de 2026.

- [Catálogo oficial](https://catalogodatos.gub.uy/dataset/conteo-vehicular-en-las-principales-avenidas-de-montevideo)
- La selección original está en `data/montevideo_seleccion.csv`.
- Las tasas por acceso están en `data/perfiles_montevideo.csv`.
- La procedencia, licencia, transformaciones y limitaciones están en `data/README.md`.

Las llegadas por segundo se simulan a partir de tasas derivadas de los sensores.
Las colas, acciones y esperas son resultados del simulador. El trabajo no evalúa
un controlador instalado en una intersección real.

## Resultados e informe

- `results/evaluacion.csv`: cada ejecución de evaluación.
- `results/resumen.csv`: medias y desvíos por escenario y controlador.
- `results/diferencias_pareadas.csv`: diferencias frente al semáforo fijo.
- `results/unidad_volumen_hora.json`: evidencia sobre la unidad de `volumen_hora`.
- `results/sensibilidad_montevideo.csv` y `results/sensibilidad_montevideo_pareadas.csv`:
  Montevideo evaluado con la lectura de cinco minutos (tasas x 12).
- `results/figures/`: gráficos.
- `report/informe.md` y `report/informe.pdf`: informe grupal.
- `docs/explicacion_codigo.md`: recorrido del código y conceptos para defenderlo.
- `GUIA_DEFENSA.md`: resumen de decisiones, cifras, puntos débiles y preguntas para la defensa oral.
- `guia_proyecto.ipynb`: notebook ejecutado que recorre el proyecto de punta a punta.
- `demo/`: simulador 3D del cruce con un episodio de cada controlador. Solo dibuja lo que
  calcula `src/entorno.py`: no cambia ningún resultado.

El DQN mejora frente al control fijo en los escenarios sintéticos evaluados.
En los perfiles de Montevideo, con la unidad publicada, gana el control por
mayor cola. Pero el archivo de enero de 2026 trae en `volumen_hora` el conteo de
cinco minutos, no vehículos por hora: solo el 6,5 % de los valores son múltiplos
de 12 y, leído así, el total mensual de Rivera coincide con el indicador oficial
de la Intendencia. Con esa lectura, y sin reentrenar, DQN obtiene la menor espera
(8,51 s, frente a 8,98 s de mayor cola y 14,65 s del fijo). El informe desarrolla
ambos resultados y sus limitaciones.

## Organización

| Carpeta | Contenido |
|---|---|
| `src/` | `entorno.py` (el cruce), `controladores.py` (fijo y mayor cola), `q_learning.py`, `dqn.py`, `datos.py` (ventanas de Montevideo) y `experimento.py` (cargar controladores y correr episodios). |
| `scripts/` | `preparar_datos.py`, `entrenar.py`, `evaluar.py`, `sensibilidad_montevideo.py`, `graficos.py`, `demo.py` e `informe_pdf.py`. |
| `tests/` | `test_proyecto.py`: 8 comprobaciones del simulador y de los agentes. |
| `configs/` | `experimento.json`: semillas, episodios e hiperparámetros. |
| `models/` | Los 10 agentes entrenados y una ficha JSON por agente con versiones y tiempo. |
| `data/` | Selección original de la Intendencia, perfiles derivados y auditoría. |
| `results/` | Resultados, registros de entrenamiento y gráficos. |
| `demo/` | Simulador 3D: `index.html`, los episodios en `datos.js` (los genera `scripts/demo.py`) y `LEEME.md`. |

El código está en castellano, como el resto del curso.
`guia_proyecto.ipynb` recorre el proyecto paso a paso (simulador, controladores,
una actualización de Q-learning, un entrenamiento chico, DQN y resultados). Solo
lee archivos del repo: los resultados oficiales salen de los scripts.

## Referencias

- Watkins y Dayan (1992). [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
- Mnih et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).
- [NumPy](https://numpy.org/doc/stable/) y [PyTorch](https://docs.pytorch.org/docs/stable/index.html).

## Licencia de los datos

Los datos de la Intendencia se distribuyen bajo la **Licencia de Datos Abiertos -
Uruguay**. La nota de origen está en `data/README.md`.
El proyecto no asigna una licencia al código.
