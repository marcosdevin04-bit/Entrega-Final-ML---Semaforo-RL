# Entrega-Final-ML---Semaforo-RL
Control adaptativo de semáforos con aprendizaje por refuerzo. Comparación de Q-learning, tiempos fijos y reglas de prioridad en una intersección simulada en Python.


# Semáforo RL 🚦

Proyecto final grupal de Machine Learning sobre control adaptativo de semáforos mediante aprendizaje por refuerzo.

## Objetivo

Investigar si un agente de Q-learning puede reducir los tiempos de espera frente a un semáforo de tiempos fijos y un controlador basado en reglas, bajo distintas condiciones de tránsito.

## El problema

Los tiempos fijos de un semáforo pueden resultar poco adecuados cuando cambia la demanda de vehículos. Este proyecto aborda el problema mediante un controlador que aprende a tomar decisiones a partir de la situación del cruce.

El trabajo utiliza una simulación propia en Python. Los resultados y las conclusiones se limitan a las condiciones representadas en ella.

## Entorno de simulación

Se considera una intersección con dos fases de circulación: norte–sur y este–oeste.

Los vehículos llegan de forma aleatoria y forman colas. El controlador decide mantener la fase actual o solicitar un cambio, respetando tiempos mínimos en verde y un intervalo de transición.

## Estrategias de control

- **Tiempos fijos:** alternancia de fases con duraciones predefinidas.
- **Control por reglas:** prioridad al sentido con mayor cantidad de vehículos esperando.
- **Q-learning:** selección de acciones mediante una política aprendida por interacción con el entorno.

Todos los controladores están sujetos a las mismas restricciones.

## Aprendizaje por refuerzo

El controlador del semáforo actúa como agente. A partir de las colas de vehículos, la fase actual y el tiempo desde el último cambio, decide si mantiene la fase o solicita una transición.

La recompensa busca penalizar la congestión y las esperas, considerando el costo de cambiar de fase.

## Evaluación

Se comparan las estrategias en escenarios de demanda equilibrada, mayor demanda en un sentido y cambios de demanda durante el episodio.

Las métricas de evaluación incluyen:

- Tiempo de espera promedio.
- Percentil 95 del tiempo de espera.
- Cantidad de vehículos que cruzan.
- Cantidad de vehículos pendientes al finalizar.
- Longitud de las colas.

La comparación utiliza varias semillas y patrones de llegada compartidos entre los controladores. Los escenarios de evaluación se separan de los utilizados para ajustar las estrategias.

## Tecnologías

- Python.
- NumPy para cálculo numérico.
- Matplotlib para visualizar resultados.

## Contexto académico

Proyecto integrador grupal de Machine Learning sobre aprendizaje por refuerzo.

**Entrega y exposición:** 26 de noviembre de 2026.
