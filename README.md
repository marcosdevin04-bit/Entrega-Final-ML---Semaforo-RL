# Semáforo RL

Control adaptativo de una intersección mediante aprendizaje por refuerzo.
Proyecto final grupal de Machine Learning.

**Integrantes:** Marcos Devinenzi, Facundo Diaz, Santiago Alba y Domingo Leiva.
**Docente:** Gustavo Vazquez. **Entrega:** 26 de noviembre de 2026.

## Objetivo

Comparar tiempos fijos, prioridad a la mayor cola, Q-learning tabular y DQN bajo
las mismas condiciones de tránsito. El caso de aplicación incluye perfiles
derivados de conteos oficiales de Montevideo y escenarios sintéticos controlados.

## Motivación y antecedentes

Elegimos estudiar aprendizaje por refuerzo y aplicarlo a un problema cotidiano de movilidad. La investigación inicial identificó antecedentes municipales de control adaptativo. Nuestro trabajo aporta una simulación y una comparación propia; sus resultados no evalúan al sistema de la Intendencia.

El aprendizaje por refuerzo es un tema nuevo para el grupo: el proyecto implica
estudiar sus fundamentos, implementarlos y explicar su funcionamiento.

La documentación y las fuentes están en [Antecedentes en Montevideo](docs/antecedentes_montevideo.md).

## Instalación

Usar Python 3.11 o 3.12. Desde la carpeta del proyecto:

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
python -m scripts.evaluate
python -m scripts.plots

# Generar una demo que se abre con doble clic, sin servidor
python -m scripts.demo
```

Abrir `results/demo.html`. Permite reproducir, pausar, cambiar la velocidad y
recorrer el episodio. Para usar los perfiles de Montevideo:

```powershell
python -m scripts.demo --scenario montevideo
```

Para entrenar nuevamente los diez agentes y reproducir el experimento completo:

```powershell
python -m scripts.train
python -m scripts.evaluate --select-baseline
python -m scripts.plots
python -m scripts.demo --gif
```

La configuración está en `configs/experiment.json`. La corrida completa usa
700 episodios de 600 segundos por agente y cinco semillas por algoritmo.
También se puede probar un entrenamiento breve con
`python -m scripts.train --algorithm q_learning --episodes 100 --seeds 0`.
**Ese comando reemplaza el modelo de la semilla 0**: hacerlo en una copia si se
quieren conservar los resultados incluidos. Los modelos guardados contienen
pesos/tablas para inferencia, no el estado necesario para reanudar entrenamiento.

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

- `results/evaluation.csv`: cada ejecución de evaluación.
- `results/summary.csv`: medias y desvíos por escenario y controlador.
- `results/paired_differences.csv`: diferencias frente al semáforo fijo.
- `results/figures/`: gráficos.
- `report/informe.md` y `report/informe.pdf`: informe grupal.
- `presentation/semaforo_rl_final.pptx`: presentación para la exposición.
- `docs/explicacion_codigo.md`: recorrido del código y conceptos para defenderlo.
- `guia_proyecto.ipynb`: notebook ejecutado que recorre el proyecto de punta a punta.

El DQN mejora frente al control fijo en los escenarios sintéticos evaluados.
En los perfiles de Montevideo gana el control por mayor cola. El informe
desarrolla ambos resultados y sus limitaciones.

## Organización

`src/` contiene simulador, controladores y agentes. `scripts/` contiene los
puntos de entrada. Las comprobaciones están separadas en `tests/`.
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
