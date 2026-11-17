# Recorrido del proyecto

## Orden para leer el código

1. `src/entorno.py`: cómo se generan llegadas, se forman colas y cruzan vehículos.
2. `src/controladores.py`: las dos reglas que sirven como comparación.
3. `src/q_learning.py`: tabla Q, discretización, exploración y actualización.
4. `scripts/entrenar.py`: bucle de episodios compartido por los agentes.
5. `src/dqn.py`: misma idea de valores Q con una red neuronal.
6. `scripts/evaluar.py`: separación de validación y prueba, comparación y métricas.
7. `scripts/graficos.py` y `scripts/demo.py`: resultados y visualización.
8. `scripts/sensibilidad_montevideo.py`: evidencia sobre la unidad de los datos y
   evaluación de Montevideo con la lectura de cinco minutos.

El notebook `guia_proyecto.ipynb` recorre lo mismo ejecutando el código paso a paso.

Los nombres de archivo y variables se mantienen estables. Los comentarios explican
las decisiones principales. Las pruebas están en un archivo separado para que el
flujo de entrenamiento se pueda leer de corrido.

## Qué hace un episodio

El entorno comienza vacío y sortea la fase inicial. Antes de ejecutar acciones
genera todas las llegadas de los siguientes 600 segundos. Cada cinco segundos el
controlador recibe una observación y elige mantener o solicitar un cambio.

Dentro de cada segundo primero llegan vehículos, después salen los habilitados
y quienes siguen esperando suman un segundo. En verde sale como máximo un
vehículo por segundo del acceso activo. Durante tres segundos de transición
no sale ninguno. El verde mínimo es 10 segundos y el límite superior 60.

La grilla de decisiones hace que algunos verdes duren 12, 17 o 22 segundos,
porque la transición ocupa tres segundos del primer bloque de cinco. El entorno
adelanta un cambio forzado cuando el siguiente bloque superaría 60 segundos.
Ese límite restringe la duración de una fase, pero no garantiza una espera máxima
para cada vehículo si una cola crece por exceso de demanda.

## Q-learning explicado

La tabla guarda dos valores por estado: uno para mantener y otro para cambiar.
La observación contiene dos colas, la fase, la edad del verde y el tiempo relativo
del episodio. Para usar una tabla agrupamos colas, edad y tiempo en intervalos.

Cuando recibe una recompensa actualiza solo la celda correspondiente a la acción
que tomó. La nueva estimación combina la recompensa con el mejor valor del
estado siguiente. Si el episodio terminó no agrega valor futuro.

Con probabilidad epsilon prueba una acción al azar. El resto del tiempo elige
la que tiene mayor valor. Durante la evaluación epsilon es cero.

## Qué cambia en DQN

En lugar de consultar una celda, pasa las cinco observaciones a una red de dos
capas ocultas de 64 unidades. La red devuelve los valores de las dos acciones.
Usa ReLU, Adam y pérdida Huber.

Dos mecanismos adicionales estabilizan el aprendizaje:

- **Replay:** guarda experiencias y toma minibatches aleatorios de 64 para entrenar.
- **Red objetivo:** una copia de la red que permanece fija entre actualizaciones.
  Se sincroniza cada 250 actualizaciones del optimizador.

Entrena una vez cada cuatro decisiones, después de reunir 1.000 experiencias.
Tiene un buffer de 30.000 transiciones y recorta la norma del gradiente a 10.
No incorpora variantes como Double DQN ni replay priorizado.

## Por qué la recompensa es negativa

Por cada decisión usamos menos la suma de las colas de sus cinco segundos,
dividida por 100. Así el agente busca reducir el tiempo total que los vehículos
pasan esperando. Dividir por 100 cambia la escala numérica, no el objetivo.
La transición ya cuesta espera, por eso no agregamos una penalización arbitraria
adicional por cambiar.

## Cómo leer las métricas

La espera observada media es el área bajo las colas dividida por todos los
vehículos que llegaron. Incluye la espera acumulada de los pendientes. Para
esos vehículos es una espera censurada: todavía no sabemos su espera final.
Por eso también informamos cuántos quedaron sin cruzar. El percentil 95 se calcula
sobre esas mismas esperas observadas y tiene la misma limitación.

Los desvíos de las tablas combinan variación entre episodios y, para los agentes,
entre semillas de entrenamiento. Las diferencias pareadas promedian primero los
cinco agentes por semilla de tránsito. Sus intervalos describen esa colección
de políticas fija, no toda la incertidumbre de entrenar un algoritmo nuevamente.

## Preguntas que deben poder responder

- ¿Por qué esto es aprendizaje por refuerzo? Las acciones cambian las colas futuras
  y el agente aprende de recompensas de su interacción, sin etiquetas de acción correcta.
- ¿Por qué no usar solamente reglas? La regla de mayor cola es una referencia muy
  fuerte y barata. DQN la supera en los escenarios sintéticos y, con la lectura
  corregida de Montevideo, por poco (0,47 s por vehículo). Con la unidad publicada,
  en cambio, la regla gana en Montevideo.
- ¿DQN siempre supera Q-learning? No existe esa garantía. Acá lo supera en varias
  condiciones. Con la unidad publicada ambos pierden frente a las reglas en
  Montevideo; con la lectura de cinco minutos ambos mejoran al fijo.
- ¿Por qué hay dos resultados para Montevideo? La columna `volumen_hora` debería ser
  un conteo de cinco minutos multiplicado por 12, pero solo el 6,5 % de sus valores
  son múltiplos de 12, y leída como veh/h da 12 veces menos tránsito en Rivera que
  el indicador oficial de la Intendencia. Lo encontramos después de la prueba, por
  eso lo presentamos como análisis de sensibilidad y no reentrenamos.
- ¿Es un MDP perfectamente observado? No afirmamos eso: la discretización pierde
  información y las tasas de llegada no forman parte de la observación. El tiempo
  ayuda a describir el horizonte, pero no elimina toda la información oculta.
- ¿Puede instalarse en una calle? Este trabajo no lo valida. Faltan geometría real,
  peatones, giros, coordinación y calibración con esperas observadas.
