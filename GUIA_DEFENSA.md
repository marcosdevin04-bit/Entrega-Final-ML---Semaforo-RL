# Guía para defender el trabajo final

Resume qué hicimos, por qué, y con qué números lo sostenemos. Todas las cifras salen de `results/` y del informe. El tema no se dio en clase, así que el docente va a querer ver que **los cuatro** entendemos aprendizaje por refuerzo, no solo el código. Esta guía ayuda, pero no reemplaza correr `guia_proyecto.ipynb` y leer `src/`.

Convenciones: las esperas son **segundos por vehículo**, promedio de 30 episodios de prueba por escenario (para los agentes, 30 episodios × 5 semillas de entrenamiento). Una diferencia **negativa** frente al fijo significa **menos espera**.

## Presentación de un minuto

> Elegimos aprendizaje por refuerzo, un tema que no se dio en el curso, y lo aplicamos a un semáforo. Programamos un simulador de un cruce con dos calles y comparamos cuatro formas de manejar la luz: tiempos fijos, una regla que le da verde a la calle con más autos, Q-learning con una tabla y DQN con una red neuronal. Todos enfrentan exactamente el mismo tránsito. En los escenarios sintéticos DQN tiene la menor espera: por ejemplo, cuando la demanda cambia a mitad del episodio baja de 14,95 a 7,66 segundos por auto frente al fijo. También usamos datos reales de los sensores de la Intendencia en Bv. Artigas y Rivera. Con la unidad publicada, ahí gana la regla simple. Pero encontramos que la columna `volumen_hora` en realidad trae conteos de cinco minutos: solo el 6,5 % de los valores son múltiplos de 12 y, leída así, coincide con el indicador oficial de la Intendencia. Con esa lectura, y sin reentrenar, DQN vuelve a tener la menor espera, aunque la regla queda muy cerca. La conclusión es que aprender una política sirve cuando hay tránsito y la demanda es desigual o cambia, pero una regla simple es una referencia muy difícil de superar.

## Reparto sugerido para la exposición

| Bloque | Contenido | Diapositivas |
|---|---|---|
| 1 | Problema, qué es aprendizaje por refuerzo, antecedentes en Montevideo | 1–3 |
| 2 | Simulador, observación, acciones y recompensa | 4 |
| 3 | Q-learning y DQN, entrenamiento y evaluación | 5, 7, 10 |
| 4 | Datos de Montevideo, resultados, unidad de `volumen_hora`, conclusión y demo | 6, 8, 9, 11, 12 y la nueva del ×12 |

Cada uno tiene que poder responder preguntas de **cualquier** bloque.

## Conceptos del tema nuevo

### Aprendizaje por refuerzo frente a lo visto en el curso

| | Supervisado (entregas 1 y 2) | Por refuerzo (este trabajo) |
|---|---|---|
| Qué recibe | Ejemplos con la respuesta correcta | Una recompensa después de actuar |
| Qué aprende | A predecir la etiqueta | Una **política**: qué acción tomar en cada estado |
| Los datos | Fijos de antemano | Dependen de las acciones del agente |
| Cómo se evalúa | Métricas sobre un conjunto de prueba | Simulando episodios con la política aprendida |

En nuestro caso: el **agente** es el semáforo; el **estado** son las dos colas, la fase, la edad del verde y el tiempo del episodio; las **acciones** son mantener (0) o pedir un cambio (1); la **recompensa** es menos la cantidad de autos esperando.

### Términos que hay que saber definir

- **Proceso de decisión de Markov (MDP):** estados, acciones, transiciones, recompensas y descuento. El siguiente estado depende solo del estado y la acción actuales.
- **Retorno descontado:** `G = r₀ + γ r₁ + γ² r₂ + …`. Usamos **γ = 0,97 por decisión**. Una decisión es cada 5 s, así que el agente "mira" del orden de 1/(1−γ) ≈ 33 decisiones, unos 2,8 minutos.
- **Función Q(s, a):** cuánto retorno se espera si en el estado *s* se toma la acción *a* y después se sigue la política.
- **Epsilon-greedy:** con probabilidad ε se elige una acción al azar (explorar); si no, la de mayor Q (explotar). ε baja de 1 a 0,05 durante el primer 75 % del entrenamiento. En la evaluación ε = 0.
- **Episodio:** 600 segundos de simulación, 120 decisiones.

### Q-learning

Regla de actualización (Watkins y Dayan, 1992):

`Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]`

- α = 0,12 es cuánto se corrige cada vez. Si el episodio terminó, no se suma el término futuro.
- **Ejemplo del notebook:** la tabla empieza en 0, la recompensa es −1,07 y el estado siguiente vale 0. Entonces `Q = 0 + 0,12 · (−1,07 + 0,97 · 0 − 0) = −0,1284`.
- La observación es continua, así que se **agrupa en intervalos** (`codificar` en `src/q_learning.py`): 7 grupos por cola, 2 fases, 7 grupos de edad y 3 tercios del episodio = **2.058 estados**, con 2 valores cada uno.

### DQN

Misma idea, pero Q(s, a) la calcula una **red neuronal 5 → 64 → 64 → 2** con ReLU (como en el lab de redes), que recibe las 5 observaciones **sin agrupar**.

- **Memoria de experiencias (replay):** se guardan las últimas 30.000 jugadas y se entrena con lotes de 64 tomados al azar. Sirve para que las muestras de un lote no estén tan correlacionadas y para reusar experiencia.
- **Red objetivo:** una copia de la red que se usa para calcular `r + γ max Q(s', a')` y se actualiza cada 250 pasos de entrenamiento. Sin ella, el objetivo se movería en cada paso y el entrenamiento sería inestable.
- **Pérdida Huber** (`smooth_l1_loss`): cuadrática para errores chicos y lineal para errores grandes, para que un error grande no dé un paso de gradiente enorme. También se recorta la norma del gradiente a 10.
- **Adam** con tasa 0,0005. Entrena cada 4 decisiones, desde que hay 1.000 jugadas guardadas.

## Cómo está armado el código

- **`src/entorno.py`:** el simulador (clase `Cruce`). `reiniciar()` sortea las llegadas antes de empezar; `avanzar(accion)` simula 5 segundos en tres pasos (decidir si cambia la fase, simular segundo a segundo, calcular la recompensa); `metrics()` resume el episodio.
- **`src/controladores.py`:** `SemaforoFijo` y `MayorCola`.
- **`src/datos.py`:** carga las ventanas de Montevideo y genera sus llegadas.
- **`src/experimento.py`:** carga los controladores guardados y corre un episodio completo (`correr_episodio`).
- **`src/q_learning.py`** y **`src/dqn.py`:** los dos agentes, con `actuar`, `actualizar` (Q-learning) o `recordar` y `aprender` (DQN), y `guardar`/`cargar`.
- **`scripts/entrenar.py`:** el bucle de entrenamiento, igual para los dos agentes.
- **`scripts/evaluar.py`:** elige el fijo en validación y compara todo en prueba.
- **`scripts/sensibilidad_montevideo.py`:** evidencia sobre la unidad y evaluación con ×12.
- **`tests/test_proyecto.py`:** 8 comprobaciones (no se pierden autos, transición, verde mínimo y máximo, llegadas independientes de las acciones, separación por fechas, actualización de Q-learning y de DQN).

## 1. Simulador

- **Un cruce, dos calles** que compiten por el verde, sin giros, peatones ni otros cruces.
- **Reglas:** verde mínimo 10 s, máximo 60 s, 3 s de transición en la que nadie cruza, 1 auto por segundo con verde. Los autos cruzan en orden de llegada.
- **Las llegadas se generan antes de decidir** (Poisson con la tasa del escenario). Así los cuatro controladores enfrentan exactamente el mismo tránsito y se pueden comparar episodio por episodio.
- **Verdes de 12, 17 o 22 segundos:** la transición ocupa 3 de los 5 segundos del primer bloque después de un cambio.

## 2. Observación, acciones y recompensa

- **Observación:** `[cola_a/40, cola_b/40, fase, edad_verde/60, tiempo/600]`. Las escalas son constantes, no se aprenden de los datos.
- **Por qué el tiempo está en la observación:** el episodio termina a los 600 s. Sin el tiempo, el mismo estado tendría valores distintos según cuánto falta.
- **Recompensa:** `−(autos esperando sumados en los 5 segundos) / 100`. Minimizar esa suma es minimizar la espera total. El /100 solo cambia la escala para que la red entrene mejor.
- **Por qué no hay penalización por cambiar:** los 3 s de transición ya cuestan espera.

## 3. Datos de Montevideo

- **Fuente:** "Conteo vehicular en las principales avenidas de Montevideo", Centro de Gestión de Movilidad, enero de 2026.
- **Accesos:** A = Bv. Artigas, de Palmar hacia Rivera (3 carriles); B = Rivera, de Ponce hacia Bv. Artigas (2 carriles).
- **Preparación:** se suman los carriles por acceso, de 06:00 a 21:55, y se exige que estén los cinco detectores.
- **Partición por fechas:** entrenamiento del 1 al 20 de enero, validación del 21 al 25, prueba del 26 al 31. Es la misma lógica que la validación temporal de la entrega 2.
- **Episodio de Montevideo:** dos registros consecutivos de 5 minutos, reproducidos sin comprimir el tiempo (600 s).

## 4. Diseño experimental

- **Fijo elegido en validación:** se probaron verdes de 10 a 60 s y ganó **15 s** (espera media de validación 11,94, contra 12,89 con 10 s y 12,79 con 20 s).
- **Entrenamiento:** 5 semillas por algoritmo × 700 episodios = 84.000 decisiones por agente, con una mezcla de equilibrado, desigual, cambiante y Montevideo. El escenario de **estrés no se usa para entrenar**: mide generalización.
- **Costo:** cada agente de Q-learning tarda unos 2,4 s en entrenar y cada DQN unos 18 s, en CPU.
- **Prueba:** 30 semillas de tránsito por escenario. Para comparar con el fijo se promedian primero las 5 versiones de cada agente en la misma semilla (para no contar el mismo tránsito 5 veces) y después se hace un **bootstrap** de 2.000 remuestreos, como en la clase de bagging.

## 5. Resultados

Espera media (segundos por auto):

| Escenario | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Equilibrado | 9,11 | 7,78 | 9,78 | **7,73** |
| Desigual | 17,03 | 8,61 | 8,62 | **7,63** |
| Cambiante | 14,95 | 8,48 | 8,93 | **7,66** |
| Estrés | 18,49 | 13,62 | 18,74 | **11,97** |
| Montevideo (unidad publicada) | 6,96 | **4,13** | 12,74 | 12,50 |

Diferencia de DQN frente al fijo, con intervalo bootstrap:

| Escenario | DQN − fijo (s) | Intervalo |
|---|---|---|
| Equilibrado | −1,38 | −1,57 a −1,20 |
| Desigual | −9,40 | −11,04 a −7,93 |
| Cambiante | −7,29 | −8,10 a −6,49 |
| Estrés | −6,52 | −8,56 a −4,98 |
| Montevideo (unidad publicada) | +5,54 | +4,65 a +6,34 |

- **DQN** gana en los cuatro escenarios sintéticos, incluido **estrés, que no vio al entrenar**. En equilibrado prácticamente empata con mayor cola (7,73 contra 7,78).
- **Q-learning** mejora al fijo en desigual y cambiante, pero empeora en equilibrado (+0,67 s) y empata en estrés. Además es el más variable entre semillas.
- **Percentil 95 en estrés:** 24,67 s con DQN, 27,01 con mayor cola, 41,82 con el fijo y 50,61 con Q-learning. Es decir, DQN no solo baja la media, también las esperas largas.
- **Montevideo con la unidad publicada:** la cola máxima promedio es de 2 o 3 autos. Los agentes cambian de 11 a 13 veces por episodio, contra 16 de la regla, y dejan esperando a los pocos autos que llegan.

## 6. Unidad de `volumen_hora` (análisis de sensibilidad)

- **Lo que dice el catálogo:** `volumen_hora` = conteo de 5 minutos × 12.
- **Prueba 1:** si fuera así, todos los valores serían múltiplos de 12. Entre los 40.473 registros positivos, **solo el 6,5 %** lo son, y los valores más frecuentes son 1, 2, 3, 4 y 5.
- **Prueba 2:** el Observatorio de Movilidad de la Intendencia informa **295.197 a 401.215 vehículos por mes** en Rivera, de Bv. Artigas a Ponce (2017–2018). Enero de 2026 en el acceso B da **26.325** leído como veh/h y **315.897** leído como conteo de 5 minutos.
- **Con ×12** (mismos modelos, ventanas y semillas, sin reentrenar):

| Montevideo | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Lectura como veh/h | 6,96 | **4,13** | 12,74 | 12,50 |
| Lectura de 5 minutos (×12) | 14,65 | 8,98 | 10,56 | **8,51** |

- DQN − fijo: **−6,14** (−8,42 a −4,12). DQN − mayor cola: **−0,47** (−0,77 a −0,18): poco, pero todo el intervalo queda por debajo de cero.
- **Por qué no reemplazamos los resultados principales:** lo encontramos después de mirar la prueba. Cambiar los datos y reentrenar en ese momento sería ajustar el experimento al resultado. Por eso lo presentamos como sensibilidad, con todo el procedimiento a la vista.

## Los puntos débiles y cómo defenderlos

Hay que conocerlos y no negar lo que es cierto.

**1. «Lo del ×12 lo buscaron porque el resultado les dio mal.»**
Es cierto que lo revisamos después de ver la prueba, porque las tasas daban un cruce casi vacío. La defensa: no cambiamos los resultados principales, no reentrenamos ni ajustamos nada, y las dos pruebas no dependen del modelo (la de los múltiplos de 12 y la del indicador oficial). Además, con ×12 la regla de mayor cola también mejora mucho frente al fijo; no es un resultado "a medida" de DQN.

**2. «Los agentes se entrenaron con los datos mal leídos.»**
Sí: el 25 % de los episodios de entrenamiento eran de Montevideo con tasas bajísimas. La prueba con ×12 mide cuánto generalizan a un tránsito local mayor, parecido al de los escenarios sintéticos. Un experimento completo requeriría volver a preparar los datos, reentrenar y reservar otra prueba.

**3. «El semáforo fijo está en desventaja.»**
Usa el mismo verde para las dos calles. Se eligió en validación entre ocho duraciones, pero un plan con tiempos distintos por calle podría ser mejor, sobre todo en desigual. Por eso la comparación importante es contra **mayor cola**, que es adaptativa y gratis.

**4. «No sabemos si DQN gana por la red o por no agrupar la observación.»**
Correcto: DQN recibe las colas exactas y Q-learning las agrupa en intervalos. Faltaría una ablación con la misma representación para separar los dos efectos. Está declarado en el informe.

**5. «No buscaron hiperparámetros y entrenaron poco.»**
Usamos valores fijos declarados en `configs/experimento.json` y no tocamos nada mirando la prueba. Entrenar es barato (unos 18 s por DQN), así que una búsqueda en validación es la mejora más directa.

**6. «El simulador es muy simple.»**
Sí: 1 auto por segundo para todo el acceso aunque Bv. Artigas tenga 3 carriles, colas que empiezan vacías, sin giros, peatones ni coordinación con otros cruces. Además los sensores miden autos que pasaron con los semáforos actuales, no la demanda que habría con otra política. El objetivo era comparar políticas en un entorno que se pueda entender y verificar, no reproducir el cruce real.

**7. «Las ventanas de prueba de Montevideo no son independientes.»**
Pueden solaparse y compartir el mismo día. Por eso los intervalos son descriptivos y no los usamos como prueba de significación.

## Preguntas para practicar

1. **¿Por qué esto es aprendizaje por refuerzo y no supervisado?** No hay una acción correcta etiquetada; el agente aprende de recompensas, y sus acciones cambian las colas que verá después.
2. **¿Qué es γ y por qué 0,97?** Cuánto pesa el futuro. Con decisiones cada 5 s, el agente tiene en cuenta unos 2–3 minutos hacia adelante, del orden de un ciclo de semáforo.
3. **¿Para qué sirve ε y por qué baja?** Al principio la tabla o la red no saben nada y hay que probar acciones; al final conviene usar lo aprendido.
4. **¿Por qué la recompensa es negativa?** Porque es un costo: menos autos esperando es mejor. Maximizar la recompensa es minimizar la espera.
5. **¿Para qué la red objetivo y el replay?** Para que el objetivo no cambie en cada paso y para que los lotes no estén correlacionados. Son lo que hace estable a DQN.
6. **¿Por qué Q-learning anda peor que DQN?** Agrupar colas y edad en intervalos pierde información, y con 84.000 decisiones hay estados que se visitan poco. No lo aislamos experimentalmente.
7. **¿Por qué los agentes pierden en Montevideo con la unidad publicada?** El cruce está casi vacío (cola máxima de 2–3 autos) y las recompensas de esos episodios son mínimas frente a las de los sintéticos, así que casi no influyen en el aprendizaje. Los agentes cambian poco y dejan esperando a los pocos autos.
8. **¿Cómo saben que `volumen_hora` es de 5 minutos?** Múltiplos de 12 y comparación con el indicador mensual de la Intendencia (sección 6).
9. **¿Cómo se aseguran de que la comparación es justa?** Mismas llegadas y misma fase inicial para todos, las reglas del cruce las aplica el entorno para todos, el fijo se elige en validación y la prueba se usa una vez.
10. **¿Recomendarían instalar DQN en un semáforo real?** No con este trabajo. Habría que calibrar el simulador con colas y esperas reales, modelar giros y peatones, y compararlo con el Full Adaptive que la Intendencia implementó en 2024 en nueve cruces de Tres Cruces.
11. **¿Qué harían con más tiempo?** Reentrenar con la unidad corregida y una prueba nueva, buscar hiperparámetros en validación, hacer la ablación de representación y probar un fijo con tiempos distintos por calle.

## Antes de la defensa

- Subir los cambios al repo (y la presentación cuando esté).
- Agregar a la presentación la diapositiva del ×12 y corregir la 9.
- Correr `python -m unittest discover -s tests -v` y abrir el simulador con `python -m scripts.demo` en la computadora de la defensa (necesita internet).
- Tener a mano `guia_proyecto.ipynb` ejecutado por si piden mostrar el código funcionando.

## Fuentes

- Informe del grupo: `report/informe.md` y `report/informe.pdf`.
- Watkins, C. J. C. H. y Dayan, P. (1992). [Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).
- Intendencia de Montevideo. [Conteo vehicular en las principales avenidas de Montevideo](https://catalogodatos.gub.uy/dataset/conteo-vehicular-en-las-principales-avenidas-de-montevideo).
- Intendencia de Montevideo, Observatorio de Movilidad. [Cantidad de vehículos en circulación por eje vial seleccionado](https://montevideo.gub.uy/sites/default/files/biblioteca/cantidaddevehiculosencirculacionporejevialseleccionado_3.pdf).
