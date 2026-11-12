# Semáforo RL

Proyecto final grupal de Machine Learning sobre control de semáforos mediante aprendizaje por refuerzo.

## Elección del tema

Elegimos y registramos aprendizaje por refuerzo como tema del proyecto. Entre las aplicaciones que conversamos estuvieron un robot de depósito, la gestión de stock y el control de una intersección. Nos decidimos por el semáforo porque combina decisiones secuenciales, un problema cercano y resultados que podemos interpretar en términos de tiempo de espera.

## Motivación

La idea surgió de una situación cotidiana: quedar detenido en un semáforo y preguntarse si los tiempos podrían organizarse mejor según el tránsito del momento. Como usuarios del tránsito de Montevideo, percibimos dificultades de coordinación y creemos que vale la pena estudiar alternativas de mejora.

Nos pareció una idea original para nuestro proyecto porque nos permite estudiar una técnica nueva para nosotros y explorar su aplicación a una necesidad de la ciudad. El trabajo implica aprender los fundamentos del aprendizaje por refuerzo, ponerlos en práctica y explicar cómo funcionan. Nos interesa su posible utilidad social: reducir esperas y aprovechar mejor el tiempo de quienes circulan. Esa motivación es el punto de partida; la utilidad de cada estrategia debe evaluarse con resultados.

## Propuesta

Modelar una intersección simplificada en Python y comparar cuatro estrategias:

- Un semáforo de tiempos fijos.
- Una regla que prioriza el acceso con mayor cola.
- Un agente de Q-learning.
- Un agente DQN.

Los controladores compartirán las mismas condiciones de circulación. La comparación considerará las esperas, los vehículos que logran cruzar y los que quedan pendientes al finalizar cada episodio.

## Antecedentes locales

La búsqueda inicial encontró antecedentes de gestión semafórica adaptativa publicados por la Intendencia de Montevideo. Las fuentes y su relación con la propuesta están documentadas en [Antecedentes en Montevideo](docs/antecedentes_montevideo.md).

## Configuración y dependencias

- `requirements.txt`: bibliotecas para simulación, entrenamiento y gráficos.
- `requirements-report.txt`: dependencia para generar el informe en PDF.
- `configs/experiment.json`: parámetros del experimento.
- `.gitignore`: exclusión de entornos virtuales, cachés y archivos temporales.

Para preparar el entorno con Python 3.11 o 3.12:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Equipo y entrega

Marcos Devinenzi, Facundo Diaz, Santiago Alba y Domingo Leiva.

Docente: Gustavo Vazquez.

Entrega y exposición: 26 de noviembre de 2026.
