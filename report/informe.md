# Control adaptativo de semáforos mediante aprendizaje por refuerzo

Informe grupal de Machine Learning

**Integrantes:** Marcos Devincenzi, Facundo Díaz, Santiago Alba y Domingo Leiva.

**Docente:** Gustavo Vazquez. **Entrega y exposición:** 26 de noviembre de 2026.

## 1. Resumen

Comparamos cuatro estrategias para controlar una intersección simplificada: tiempos fijos, prioridad a la mayor cola, Q-learning tabular y Deep Q-Network (DQN). Construimos un simulador discreto en Python y evaluamos cada estrategia con las mismas llegadas de vehículos y fase inicial. Además de condiciones sintéticas controladas, incorporamos perfiles derivados de mediciones oficiales del Centro de Gestión de Movilidad de Montevideo.

Entrenamos cinco agentes de cada algoritmo durante 700 episodios y evaluamos 30 episodios por escenario. DQN reduce la espera observada frente al semáforo fijo en los cuatro escenarios sintéticos. En los perfiles de Montevideo, en cambio, la regla de mayor cola obtiene 4,13 segundos por vehículo, frente a 6,96 del fijo, 12,74 de Q-learning y 12,50 de DQN. Con esa lectura de los datos, los resultados no justifican recomendar un agente aprendido para el caso local.

Un análisis posterior del campo volumen_hora sugiere que el archivo podría contener conteos de cinco minutos y no vehículos por hora. Con esa hipótesis, y sin reentrenar ni ajustar nada, DQN obtiene 8,51 segundos por vehículo, frente a 8,98 de mayor cola y 14,65 del fijo. Lo presentamos como análisis de sensibilidad, no como un resultado confirmado.

La contribución del trabajo es comparar las políticas con un protocolo reproducible y estudiar cómo cambia su desempeño con la demanda. Las colas, esperas y acciones provienen de la simulación. No realizamos una evaluación de campo ni una reconstrucción completa de una intersección real.

## 2. Problema y objetivo

Un controlador de tiempos fijos mantiene su programación aunque cambien las colas. Un agente de aprendizaje por refuerzo puede ajustar sus decisiones según lo que observa, pero requiere entrenamiento y puede fallar fuera de las condiciones aprendidas.

Nuestra pregunta es si aprender una política reduce la espera frente a dos alternativas simples y qué costo tiene pasar de una tabla de valores a una red neuronal. La hipótesis es que la adaptación puede aportar cuando la demanda es desigual o cambia durante el episodio. No suponemos que una política aprendida deba superar a una regla en todas las condiciones.

El alcance comprende dos accesos que compiten por el verde, sin giros, peatones ni coordinación con otros cruces. El producto es un proyecto ejecutable en Python, acompañado de datos, modelos, resultados y una demostración.

<!-- pagebreak -->

## 3. Motivación y antecedentes en Montevideo

Elegimos aprendizaje por refuerzo como un tema nuevo para el grupo. El proyecto requiere estudiar sus fundamentos, implementarlos y explicarlos. La idea del semáforo nos interesó por una experiencia cotidiana: esperar en una intersección y preguntarnos si las fases podrían ajustarse mejor al tránsito. Esa percepción motiva la pregunta, pero no es una evaluación técnica de la red municipal.

### Gestión semafórica existente

La Intendencia describe al Centro de Gestión de Movilidad como una infraestructura de gestión del tránsito en tiempo real con semáforos centralizables y cámaras de conteo y monitoreo [7].

Una comunicación municipal del 16 de agosto de 2024 documenta la implementación de Full Adaptive en nueve intersecciones de Tres Cruces, con actualización del flujo cada tres segundos y ajuste de los tiempos según la demanda [8]. La publicación no menciona Q-learning ni DQN, y describe un antecedente puntual: no dice cuál es la cobertura actual.

### Aporte del proyecto

La originalidad que encontramos para el trabajo está en construir una aplicación propia y estudiar su comportamiento: un simulador legible, perfiles derivados de datos locales y una comparación reproducible de estrategias. El control semafórico adaptativo ya tiene antecedentes en Montevideo, y por eso lo incluimos en el marco del problema.

Nuestro experimento compara cuatro políticas dentro del simulador. No disponemos de la implementación ni de resultados operativos de Full Adaptive para evaluarlo con el mismo protocolo, así que una mejora frente al fijo simulado no dice nada sobre el sistema municipal. El objetivo académico es aprender y explicar cuándo aporta cada estrategia dentro de las condiciones representadas.

<!-- pagebreak -->

## 4. Marco teórico

### Aprendizaje por refuerzo y proceso de decisión

Un agente elige acciones, observa consecuencias y recibe recompensas. El objetivo es maximizar el retorno descontado: G_t = r_t + gamma r_(t+1) + gamma² r_(t+2) + ... El descuento gamma controla el peso de las consecuencias futuras.

El modelo de referencia es un proceso de decisión de Markov, definido por estados, acciones, transiciones, recompensas y descuento. Una política determina qué acción tomar. La función Q(s,a) estima el retorno de ejecutar una acción y continuar con una política. En este proyecto las transiciones se ejecutan en un simulador, sin que el agente conozca una fórmula para anticiparlas.

### Q-learning tabular

Q-learning actualiza el valor de una pareja estado-acción a partir de la recompensa y una estimación del mejor valor siguiente [1]:

Q(s,a) = Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]

Aquí alpha es la tasa de aprendizaje. Al terminar el episodio el término futuro vale cero. Usamos exploración epsilon-greedy: con probabilidad epsilon elegimos una acción aleatoria y, en los demás casos, la de mayor valor. La representación tabular requiere agrupar las observaciones en estados discretos. Esa agregación reduce el tamaño de la tabla, a costa de perder información.

### Deep Q-Network

DQN representa Q mediante una red neuronal que devuelve un valor por acción. Para estabilizar el entrenamiento utiliza un buffer de experiencias y una red objetivo [2]. Muestreamos transiciones del buffer en minibatches y calculamos un objetivo con la copia de la red. Actualizamos la red principal mediante gradiente y sincronizamos la copia periódicamente.

Implementamos la red con PyTorch, con pérdida Huber y el optimizador Adam [6]. La primera reduce el efecto de errores temporales grandes; el segundo ajusta los parámetros de la red. DQN puede compartir información entre observaciones parecidas, pero agrega decisiones de arquitectura, optimización y entrenamiento. Una mayor capacidad no garantiza una mejor política.

### Relación con el caso

El semáforo es el agente. Sus acciones alteran las colas que observará después, por lo que se trata de un problema secuencial. No contamos con etiquetas de la acción correcta. El entorno recompensa la reducción del tiempo acumulado en cola.

No invocamos una garantía de convergencia óptima: usamos un presupuesto finito, discretización y una demanda parcialmente observada. Las tasas de llegada no están en la observación y la agregación de estados pierde información. Las conclusiones son empíricas para las políticas y escenarios evaluados.

<!-- pagebreak -->

## 5. Simulación y formulación del problema

Cada episodio dura 600 segundos y comienza con colas vacías. Sorteamos la fase inicial con una semilla independiente de las llegadas. Generamos las llegadas de cada segundo con el generador aleatorio de NumPy [5] antes de elegir acciones, de modo que dos controladores puedan enfrentar exactamente el mismo tránsito.

*Tabla 1. Parámetros del entorno, iguales para los cuatro controladores.*

| Parámetro | Valor |
|---|---|
| Paso interno | 1 segundo |
| Intervalo de decisión | 5 segundos |
| Verde mínimo | 10 segundos |
| Límite superior de verde | 60 segundos |
| Transición sin circulación | 3 segundos |
| Capacidad de salida por acceso activo | 1 vehículo por segundo |
| Capacidad de almacenamiento de las colas | Sin límite |

En cada segundo agregamos vehículos, retiramos los habilitados y sumamos un segundo de espera a quienes permanecen en cola. Atendemos por orden de llegada. Solo puede circular un acceso por vez. Las restricciones se aplican en el entorno y son iguales para todos los controladores.

La transición ocupa parte del intervalo de decisión: después de cambiar, el primer verde efectivo dura dos segundos dentro de ese bloque. Por eso algunos verdes terminan en 12, 17 o 22 segundos. Forzamos un cambio si mantener el siguiente bloque superaría el límite de 60 segundos. El máximo de verde evita mantener indefinidamente una fase, pero no garantiza una espera máxima individual bajo congestión.

### Observaciones y acciones

La observación tiene cinco componentes: tamaño de ambas colas, fase vigente, edad de la fase y tiempo relativo del episodio. Dividimos las colas por 40, la edad por 60 y el tiempo por 600. Estas escalas son constantes fijadas por nosotros, no parámetros aprendidos con los datos de prueba. A DQN no le recortamos las colas, así que alguna entrada puede superar 1.

Q-learning agrupa cada cola con bordes 0, 2, 5, 10, 20 y 40 vehículos (siete rangos). Agrupa la edad con bordes 9, 19, 29, 39, 49 y 59 segundos, y el tiempo en tres partes. Con las dos fases, la tabla tiene 7 × 7 × 7 × 3 × 2 = 2.058 estados agregados y dos valores por estado. DQN recibe las cinco variables normalizadas sin esa discretización.

La acción 0 mantiene la fase y la 1 solicita un cambio. Una solicitud anterior al verde mínimo no produce cambio. Ambos agentes tienen las mismas acciones y restricciones.

### Recompensa

La recompensa de una decisión es menos la suma de las colas durante sus segundos internos, dividida por 100. Penaliza el tiempo total de espera. La transición ya genera un costo al impedir salidas, por lo que no añadimos una penalización arbitraria por cambiar. La evaluación final mide esperas sin descuento, mientras que el entrenamiento utiliza gamma = 0,97 por decisión. Son objetivos relacionados, pero no idénticos.

<!-- pagebreak -->

## 6. Datos oficiales de Montevideo

Utilizamos el conjunto «Conteo vehicular en las principales avenidas de Montevideo», publicado por la Intendencia de Montevideo, Departamento de Movilidad, Centro de Gestión de Movilidad [3]. El catálogo describe mediciones por carril cada cinco minutos mediante analítica de video.

Descargamos el recurso de enero de 2026 el 12 de noviembre de 2026. Su ZIP contiene 8.425.612 registros. Conservamos 44.578 filas originales de dos aproximaciones al entorno de Bulevar Artigas y Rivera, sin alterar sus valores. La selección ocupa mucho menos que el archivo mensual completo y queda incluida en el repositorio.

*Tabla 2. Accesos seleccionados y detectores.*

| Acceso | Ubicación descrita por el CSV | Detectores |
|---|---|---|
| A | Bv. Artigas, desde Palmar hacia Rivera | 701411, 701412, 701413 |
| B | Rivera, desde Ponce hacia Bv. Artigas | 701404, 701405 |

Sumamos los carriles por acceso e instante. Requerimos los cinco detectores presentes y conservamos horarios de 06:00 a 21:55. En los resultados principales dividimos el campo publicado volumen_hora entre 3.600 para obtener intensidades en vehículos por segundo. Si hubiera duplicados por detector e instante, los promediaríamos antes de sumar; no aparecieron en la selección utilizada.

*Tabla 3. Partición temporal de los registros derivados.*

| Partición temporal | Fechas de enero | Registros derivados |
|---|---|---|
| Entrenamiento | 1 al 20 | 3.835 |
| Validación | 21 al 25 | 954 |
| Prueba | 26 al 31 | 1.152 |

Formamos ventanas de dos registros consecutivos separados por cinco minutos. Cada ventana representa dos bloques de 300 segundos, correspondientes a los intervalos previos a sus registros. Generamos llegadas de Poisson con las intensidades de cada bloque. No comprimimos un día completo dentro de diez minutos.

### Diferencia entre medición y simulación

Las mediciones son reales, pero la secuencia individual de llegadas, las colas y las esperas son simuladas. Los sensores registran circulación observada, no la demanda que existiría bajo otra política. La geometría completa del cruce y los demás movimientos tampoco se reconstruyen.

El recurso contiene volumen_hora, sin la columna volumen que menciona la descripción general. El TXT y la página también difieren al describir si el valor resume la última hora o extrapola cinco minutos. En los resultados principales respetamos la unidad horaria publicada. Esta incertidumbre y las tasas bajas resultantes condicionan la interpretación local; en la sección 11 analizamos la hipótesis de que el valor corresponda al conteo de los cinco minutos.

La nota de origen, la URL de descarga, la licencia de Datos Abiertos de Uruguay [4], los metadatos originales y el hash SHA-256 están en data/README.md y data/auditoria_datos.json. La selección y el procesamiento son modificaciones nuestras, no del proveedor.

<!-- pagebreak -->

## 7. Diseño experimental

### Controladores de referencia

El controlador fijo usa la misma duración para ambos accesos. Elegimos entre 10, 15, 20, 25, 30, 40, 50 y 60 segundos con 20 semillas por escenario de validación. El criterio es la media de espera observada sobre equilibrado, desigual, cambiante y perfiles locales de validación. La duración seleccionada es 15 segundos. Es una referencia ajustada, aunque no un plan vial optimizado con tiempos distintos por acceso.

La regla de mayor cola solicita un cambio cuando el otro acceso tiene más vehículos. En empate mantiene la fase. Ambas estrategias pasan por las restricciones del simulador.

### Entrenamiento de los agentes

Cada algoritmo usa cinco semillas, de 0 a 4. Cada agente entrena 700 episodios de 120 decisiones: 84.000 interacciones. Ambos reciben la misma secuencia de escenarios y llegadas por semilla. La mezcla incluye equilibrado, desigual, cambiante y ventanas locales de entrenamiento, con probabilidades iguales.

*Tabla 4. Configuración de los agentes.*

| Parámetro | Q-learning | DQN |
|---|---|---|
| Descuento gamma | 0,97 | 0,97 |
| Tasa de aprendizaje | 0,12 | 0,0005 con Adam |
| Exploración epsilon | De 1,00 a 0,05 | De 1,00 a 0,05 |
| Reducción de la exploración | Primer 75 % del entrenamiento | Primer 75 % del entrenamiento |
| Representación | Tabla de 2.058 estados | Red 5-64-64-2 con ReLU (4.674 parámetros) |
| Memoria de experiencias y minibatch | No utiliza | 30.000 transiciones, minibatch de 64 |
| Comienzo de las actualizaciones | Desde la primera transición | Desde 1.000 transiciones |
| Frecuencia de actualización | Cada decisión | Cada 4 decisiones |
| Red objetivo | No utiliza | Se copia cada 250 actualizaciones |

Usamos los parámetros declarados, sin una búsqueda exhaustiva de hiperparámetros. Guardamos la política final de cada entrenamiento. No seleccionamos el mejor agente mirando la prueba ni volvimos a ajustar después de observar esos resultados. Los modelos se ejecutan en CPU.

### Escenarios y evaluación final

*Tabla 5. Intensidades de los escenarios sintéticos, en vehículos por segundo.*

| Escenario | Acceso A | Acceso B | Uso |
|---|---|---|---|
| Equilibrado | 0,23 | 0,23 | Entrenamiento y evaluación |
| Desigual | 0,40 | 0,10 | Entrenamiento y evaluación |
| Cambiante | 0,40 y desde 300 s 0,10 | 0,10 y desde 300 s 0,40 | Entrenamiento y evaluación |
| Estrés | 0,42 | 0,32 | Solo evaluación |

El caso Montevideo usa 30 ventanas de las fechas de prueba, del 26 al 31 de enero.

Evaluamos 30 semillas de tránsito por escenario. Cada agente entrenado enfrenta esas 30 realizaciones sin exploración. Hay 150 ejecuciones por algoritmo aprendido y escenario, y 30 por referencia. No interpretamos las 150 como observaciones de tránsito independientes.

## 8. Resultados de espera observada

Definimos la espera observada media como la suma de todos los segundos en cola dividida por los vehículos que llegaron. Incluye a quienes siguen esperando al final. Para ellos, la espera está censurada por el horizonte: no equivale a su espera final hasta cruzar.

La tabla muestra media y desvío estándar en segundos por vehículo. Para los agentes, el desvío combina variación entre episodios y entre semillas de entrenamiento. Describe la dispersión; no es un intervalo de confianza de la media.

*Tabla 6. Espera observada media y desvío estándar, en segundos por vehículo.*

| Escenario | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Equilibrado | 9,11 ± 0,47 | 7,78 ± 0,57 | 9,78 ± 2,69 | 7,73 ± 0,51 |
| Desigual | 17,03 ± 4,76 | 8,61 ± 0,87 | 8,62 ± 1,08 | 7,63 ± 0,79 |
| Cambiante | 14,95 ± 2,64 | 8,48 ± 0,69 | 8,93 ± 2,41 | 7,66 ± 0,70 |
| Estrés | 18,49 ± 5,92 | 13,62 ± 2,65 | 18,74 ± 6,48 | 11,97 ± 1,89 |
| Montevideo | 6,96 ± 1,42 | 4,13 ± 1,05 | 12,74 ± 3,66 | 12,50 ± 3,87 |

![Figura 1. Espera observada media y desvío estándar por escenario y controlador; Montevideo con la lectura como veh/h](../results/figures/espera_media.png)

DQN obtiene la menor media entre los cuatro controladores en los cuatro escenarios sintéticos. Frente al fijo reduce la espera un 15,2 % en equilibrado, 55,2 % en desigual, 48,7 % en cambiante y 35,3 % en estrés. Frente a mayor cola, las reducciones son 0,6 %, 11,4 %, 9,7 % y 12,2 %: en equilibrado la diferencia es despreciable. Estos porcentajes comparan las medias del experimento, no mejoras garantizadas fuera de él.

Q-learning mejora frente al fijo en desigual y cambiante, empeora en equilibrado y prácticamente empata en estrés. Su desempeño depende del régimen de demanda y presenta más variabilidad. La discretización y la visita finita a estados son explicaciones posibles, aunque no las aislamos experimentalmente.

En Montevideo la regla de mayor cola obtiene la menor espera, y los dos agentes casi duplican la espera del fijo. Pasar de la tabla a la red no resuelve el caso local con este entrenamiento. Con esta lectura de los datos, recomendamos la regla como referencia para ese régimen simulado y dejamos registrado el resultado negativo del aprendizaje. Como mostramos en la sección 11, este resultado depende de la unidad del dato.

## 9. Esperas altas y vehículos pendientes

También medimos el percentil 95 de las esperas observadas. Incluye las esperas acumuladas de los pendientes y hereda su censura. La tabla informa el promedio de los percentiles por ejecución; no es el percentil de todos los vehículos mezclados.

*Tabla 7. Promedio del percentil 95 de la espera observada por ejecución, en segundos.*

| Escenario | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Equilibrado | 22,14 | 17,63 | 28,65 | 18,19 |
| Desigual | 36,77 | 18,43 | 23,66 | 18,98 |
| Cambiante | 34,12 | 18,35 | 25,99 | 19,43 |
| Estrés | 41,82 | 27,01 | 50,61 | 24,67 |
| Montevideo | 20,45 | 12,34 | 44,61 | 45,25 |

En estrés, DQN combina una media menor con un percentil 95 de 24,67 segundos, frente a 41,82 del fijo y 50,61 de Q-learning. En equilibrado, desigual y cambiante, mayor cola tiene un percentil 95 algo menor que DQN aunque su media sea mayor: las dos métricas no ordenan igual a los controladores. En Montevideo, mayor cola obtiene 12,34 segundos frente a 45,25 de DQN; el agente no solo tiene una media peor en ese caso, también concentra esperas altas más grandes.

La cantidad media de vehículos pendientes al finalizar complementa la espera, para no ocultar colas sin atender.

*Tabla 8. Cantidad media de vehículos pendientes al finalizar el episodio.*

| Escenario | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Equilibrado | 4,00 | 3,13 | 3,69 | 2,79 |
| Desigual | 10,37 | 4,83 | 4,04 | 3,86 |
| Cambiante | 8,20 | 4,67 | 5,07 | 3,74 |
| Estrés | 17,87 | 11,53 | 15,62 | 9,29 |
| Montevideo | 0,10 | 0,03 | 0,56 | 0,58 |

En estrés quedan 9,29 vehículos con DQN frente a 17,87 con el fijo. En Montevideo las cantidades son pequeñas para todos, pero Q-learning y DQN dejan más pendientes que las referencias. Que haya pocos vehículos no hace irrelevante su espera: con baja demanda, mantener el verde cuando hay alguien esperando del otro lado genera demoras evitables.

Los CSV también incluyen vehículos que cruzaron, máxima cola y cantidad de cambios. No reducimos la decisión a una única cifra de recompensa.

<!-- pagebreak -->

## 10. Aprendizaje e incertidumbre

![Figura 2. Espera durante el entrenamiento, con media móvil de 40 episodios; líneas finas por semilla y línea gruesa promedio](../results/figures/aprendizaje.png)

Los episodios de entrenamiento tienen exploración y demandas distintas, así que la figura no es una curva de evaluación controlada ni prueba una mejora monótona. La comparación final, sin exploración, está en las tablas anteriores.

Para comparar con el fijo, primero promediamos los cinco agentes de cada algoritmo por semilla de tránsito. Después calculamos diferencias pareadas y un bootstrap de 2.000 remuestreos de esas semillas. Así evitamos contar cinco veces un mismo episodio del fijo como si aportara información nueva.

*Tabla 9. Diferencia pareada de DQN frente al fijo, en segundos por vehículo, con intervalo bootstrap descriptivo del 95 %.*

| Escenario | Diferencia media | Límite inferior | Límite superior |
|---|---|---|---|
| Equilibrado | -1,38 | -1,57 | -1,20 |
| Desigual | -9,40 | -11,04 | -7,93 |
| Cambiante | -7,29 | -8,10 | -6,49 |
| Estrés | -6,52 | -8,56 | -4,98 |
| Montevideo | +5,54 | +4,65 | +6,34 |

El signo negativo indica menor espera que el fijo y el positivo, mayor espera. En los cuatro escenarios sintéticos todo el intervalo queda por debajo de cero; en Montevideo, con la lectura como veh/h, queda por encima. Las diferencias del resto de los controladores están en results/diferencias_pareadas.csv.

Los intervalos son condicionales a las cinco políticas entrenadas. No capturan toda la incertidumbre de repetir el entrenamiento y, en Montevideo, las ventanas pueden solaparse y compartir condiciones del mismo día. Por eso no los interpretamos como intervalos sobre una población de días independientes ni como pruebas definitivas de significación.

### Costo de entrenamiento

Ambos algoritmos reciben la misma cantidad de interacciones, pero no el mismo costo computacional: la red agrega inferencias, memoria de experiencias y actualizaciones por gradiente. En el equipo donde corrimos el experimento, entrenar un agente tabular llevó en promedio 2,42 segundos y uno DQN 18,35 segundos, unas 7,6 veces más. Los tiempos y las versiones de cada modelo quedan en models/*.json; en otra computadora pueden ser distintos.

## 11. Sensibilidad a la unidad de volumen_hora

Las tasas de Montevideo resultaron unas diez veces menores que las de los escenarios sintéticos, lo que nos llevó a revisar la unidad del campo después de la evaluación. El catálogo describe volumen_hora como la cantidad de vehículos de los cinco minutos previos multiplicada por 12 [3], mientras que el TXT del recurso lo presenta como la cantidad de vehículos detectados en la última hora. Hicimos dos comprobaciones sobre las 44.578 filas originales.

### Evidencia sobre la unidad

Si el valor fuera un conteo de cinco minutos multiplicado por 12, todos los registros serían múltiplos de 12. Entre los 40.473 registros positivos solo lo son 2.611, el 6,45 %, cerca del 8,3 % que se esperaría si los valores no tuvieran relación con el 12. Además, los valores más frecuentes son 1, 2, 3, 4 y 5. Esto descarta que el archivo aplique esa multiplicación, pero por sí solo no distingue entre un conteo de cinco minutos y un total horario.

La segunda comprobación compara el total mensual con un indicador de la Intendencia. El Observatorio de Movilidad informa entre 295.197 y 401.215 vehículos por mes en Av. Rivera, tramo de Bv. Artigas a Ponce, entre 2017 y 2018 [9]. Es la misma cuadra del acceso B, pero en el sentido opuesto y en otros años.

*Tabla 10. Total de enero de 2026 en el acceso B según cada lectura, frente al indicador de la Intendencia.*

| Lectura | Vehículos |
|---|---|
| Como veh/h, usada en los resultados principales | 26.325 |
| Como conteo de cinco minutos | 315.897 |
| Indicador de la Intendencia para la misma cuadra (2017-2018) | 295.197 a 401.215 por mes |

La lectura de cinco minutos queda en el mismo orden de magnitud que el indicador; la lectura horaria queda más de diez veces por debajo. Como el indicador es de otros años y del sentido contrario, la coincidencia respalda la hipótesis, pero no la confirma. Bajo esa lectura, las tasas medias de prueba pasan de 0,027 y 0,016 a 0,33 y 0,19 veh/s, un rango comparable al de los escenarios sintéticos.

### Resultados con la lectura de cinco minutos

Repetimos la prueba de Montevideo multiplicando las tasas por 12. Usamos los mismos modelos, ventanas de prueba y semillas de tránsito que en la evaluación principal, y no reentrenamos ni ajustamos ningún controlador. Con factor 1 el procedimiento reproduce exactamente las diferencias de results/diferencias_pareadas.csv.

*Tabla 11. Espera observada media en Montevideo con ambas lecturas, en segundos por vehículo.*

| Lectura | Fijo | Mayor cola | Q-learning | DQN |
|---|---|---|---|---|
| Como veh/h | 6,96 | 4,13 | 12,74 | 12,50 |
| Como conteo de cinco minutos | 14,65 | 8,98 | 10,56 | 8,51 |

Con la lectura de cinco minutos, los tres controladores adaptativos mejoran al fijo. La diferencia pareada de DQN frente al fijo es -6,14 segundos, con intervalo descriptivo de -8,42 a -4,12. Frente a mayor cola, DQN espera 0,47 segundos menos por vehículo, con intervalo de -0,77 a -0,18: una ventaja pequeña. El percentil 95 baja de 33,77 segundos con el fijo a 19,38 con mayor cola y 19,23 con DQN.

### Cómo interpretarlo

Este análisis se hizo después de observar la prueba, así que no reemplaza a los resultados principales: lo presentamos como sensibilidad. Además, el fijo conserva el umbral de 15 segundos elegido con la lectura original, y los agentes se entrenaron con las tasas locales bajas, de modo que la prueba con factor 12 evalúa su generalización a un régimen con más tránsito. Un experimento completo requeriría confirmar la unidad con el proveedor, volver a preparar los perfiles, reentrenar y reservar una prueba nueva.

La conclusión práctica es que el orden de los controladores en Montevideo depende de la escala de la demanda. Si la hipótesis de cinco minutos es correcta, el resultado negativo se explicaría, al menos en parte, por una demanda simulada casi nula; con más tránsito el aprendizaje vuelve a ser competitivo, aunque la regla de mayor cola queda muy cerca. Los resultados se reproducen con scripts.sensibilidad_montevideo.

## 12. Discusión y limitaciones

La adaptación aporta en demanda desigual y cambiante, pero una regla simple también captura gran parte de esa mejora. DQN resulta competitivo en los escenarios sintéticos y generaliza al escenario de estrés reservado. El experimento no aísla si su ventaja proviene de la red o de recibir colas continuas, porque la comparación también cambia la discretización. Una ablación con la misma representación permitiría separar esos efectos.

En los perfiles locales, los agentes realizan menos cambios que mayor cola y presentan esperas mayores. Una hipótesis es que la mezcla de entrenamiento, dominada en magnitud de recompensa por escenarios de más tránsito, no prioriza reducir colas locales pequeñas. Otra es que el descuento favorece mantener la fase cuando cambiar cuesta tres segundos. No demostramos esas hipótesis; harían falta experimentos nuevos con datos de desarrollo y una prueba reservada.

### Decisión metodológica más frágil

La decisión más frágil es usar la suma de volumen_hora de los sensores como tasa exógena de llegada. Son mediciones de circulación bajo los semáforos existentes, no demanda independiente de sus acciones. Además, las descripciones del recurso no coinciden entre sí y los perfiles derivados son de baja demanda. Si la interpretación de la unidad o la cobertura fuera incorrecta, cambiarían la intensidad simulada y las cifras de espera de Montevideo, incluida la recomendación del controlador.

La sección 11 sugiere que la unidad usada en los resultados principales podría ser incorrecta y muestra que, en ese caso, la recomendación cambia. Mantuvimos los resultados principales para no modificar el protocolo después de observar la prueba, y presentamos ambas lecturas. Falta confirmar la semántica del campo con el proveedor y medir llegadas, colas y salidas en el cruce.

### Otros límites concretos

La descarga de un vehículo por segundo es una capacidad agregada supuesta, aunque los datos suman varios carriles, y no está calibrada con tiempos reales de descarga. Empezar con colas vacías favorece episodios cortos sin congestión acumulada previa. No modelamos bloqueo aguas abajo, giros, peatones ni propagación entre cruces.

El semáforo fijo comparte tiempos entre accesos y podría mejorar con un plan asimétrico. No agotamos los ajustes de hiperparámetros de Q-learning ni de DQN. Cinco semillas permiten observar variabilidad, pero no caracterizar todas las políticas posibles. El conjunto local corresponde a enero, por lo que no representa todo el año.

## 13. Conclusión

Con el presupuesto y el simulador utilizados, DQN reduce la espera frente al control fijo en las condiciones sintéticas, y Q-learning lo hace en dos de esos escenarios. En los perfiles derivados de Montevideo, con la unidad publicada, gana la regla de mayor cola. Con la lectura de cinco minutos, que sugieren los propios datos y es consistente con un indicador oficial, DQN obtiene la menor espera por un margen pequeño frente a esa regla; confirmarlo requiere un experimento nuevo. La elección de controlador depende del régimen de demanda, de la representación y del costo de entrenamiento.

El resultado respalda usar aprendizaje por refuerzo como una alternativa más a comparar, sin asumir que sea superior de por sí. Antes de una aplicación vial sería necesario validar la interpretación de los datos, calibrar el simulador y ampliar la evaluación con observaciones reales de espera.

## 14. Reproducibilidad y entrega

El código está en archivos Python. El notebook guia_proyecto.ipynb recorre el proyecto paso a paso, pero los resultados oficiales salen de los scripts. Las rutas se resuelven desde la carpeta del proyecto y no dependen del nombre de usuario. Los CSV reducidos y los diez modelos se incluyen para ejecutar la comparación sin descargar nuevamente el archivo mensual.

El flujo completo es: instalar requirements.txt, ejecutar las pruebas con unittest, entrenar con scripts.entrenar, evaluar con scripts.evaluar --elegir-fijo, generar gráficos con scripts.graficos, crear la demostración con scripts.demo y repetir el análisis de la unidad con scripts.sensibilidad_montevideo. El README incluye los comandos exactos.

El archivo configs/experimento.json declara semillas, episodios e hiperparámetros. Los registros de entrenamiento, selección del fijo y evaluación quedan en results/. Cada modelo tiene una ficha JSON con versiones y tiempo de entrenamiento. Los modelos guardados sirven para inferencia, no para reanudar exactamente el optimizador o la memoria de experiencias.

Las ocho pruebas automáticas verifican la conservación de vehículos y esperas, la transición sin circulación, el verde mínimo y máximo, la independencia de las llegadas respecto de las acciones, los episodios vacíos, la separación temporal, la actualización terminal de Q-learning y la actualización y el guardado de DQN.

La demostración es un simulador 3D en el navegador (scripts.demo y la carpeta demo/). Reproduce un episodio por escenario con el mismo tránsito para los cuatro controladores: cada vehículo llega y cruza en los segundos que calculó el simulador en Python. El navegador no entrena ni decide nada; solo dibuja esas trazas.

## Referencias

[1] Watkins, C. J. C. H. y Dayan, P. (1992). Q-learning. Machine Learning, 8, 279-292. https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf

[2] Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529-533. https://doi.org/10.1038/nature14236

[3] Intendencia de Montevideo, Departamento de Movilidad, Centro de Gestión de Movilidad. Conteo vehicular en las principales avenidas de Montevideo. Recurso de enero de 2026, consultado el 12/11/2026. https://catalogodatos.gub.uy/dataset/conteo-vehicular-en-las-principales-avenidas-de-montevideo

[4] Agencia de Gobierno Electrónico y Sociedad de la Información y del Conocimiento. Licencia de Datos Abiertos de Uruguay, versión 0.1. https://www.gub.uy/agencia-gobierno-electronico-sociedad-informacion-conocimiento/sites/agencia-gobierno-electronico-sociedad-informacion-conocimiento/files/documentos/publicaciones/licencia_de_datos_abiertos_0.pdf

[5] NumPy. Documentación del generador aleatorio. https://numpy.org/doc/stable/reference/random/generator.html

[6] PyTorch. Documentación de redes neuronales, Adam y SmoothL1Loss. https://docs.pytorch.org/docs/stable/index.html

[7] Intendencia de Montevideo. Centro de Gestión de Movilidad. Consultado el 12/11/2026. https://montevideo.gub.uy/tipo/area-tematica/movilidad/gestion-de-la-movilidad/centro-de-gestion-de-movilidad

[8] Intendencia de Montevideo. Implementación del Proyecto Tránsito Adaptativo. Publicado el 16/08/2024, actualizado el 19/08/2024. Consultado el 12/11/2026. https://montevideo.gub.uy/noticias/movilidad-y-transporte/implementacion-del-proyecto-transito-adaptativo

[9] Intendencia de Montevideo, Observatorio de Movilidad. Cantidad de vehículos en circulación por eje vial seleccionado (indicador INEX0105), enero de 2017 a diciembre de 2018. https://montevideo.gub.uy/sites/default/files/biblioteca/cantidaddevehiculosencirculacionporejevialseleccionado_3.pdf
