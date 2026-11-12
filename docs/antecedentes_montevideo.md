# Antecedentes de gestión semafórica en Montevideo

Consulta realizada el 12 de noviembre de 2026.

## Qué encontramos

La Intendencia describe al Centro de Gestión de Movilidad como un sistema de gestión del tránsito en tiempo real, apoyado en controladores de semáforos, cámaras de conteo y monitoreo y paneles informativos.

En una publicación del 16 de agosto de 2024, la IM informó la implementación de Full Adaptive en nueve intersecciones de Tres Cruces. Según esa comunicación, el sistema recibe información del flujo cada tres segundos y adapta los tiempos semafóricos a la demanda, contemplando el cruce peatonal.

La publicación consultada no identifica Q-learning ni DQN como algoritmos utilizados. La existencia de control adaptativo está documentada; no encontramos en estas fuentes una especificación que permita equipararlo con los métodos concretos de nuestro proyecto. La nota de 2024 describe la situación de esa fecha y no permite establecer por sí sola toda la cobertura actual.

## Relación con nuestra propuesta

Este antecedente confirma que el problema tiene una aplicación local. El aporte que buscamos para el trabajo consiste en construir una simulación comprensible y reproducible, comparar métodos con las mismas condiciones y discutir cuándo aprender una política aporta frente a reglas simples.

La originalidad que vimos al elegir el tema corresponde a nuestro enfoque de aplicación y experimentación. La investigación inicial también nos permitió reconocer trabajo previo en la ciudad y ubicar mejor nuestra propuesta.

La percepción del grupo sobre posibles problemas de coordinación motiva la pregunta de investigación. Para evaluar el funcionamiento del sistema municipal real se necesitarían otros datos y un estudio específico.

## Fuentes oficiales

- Intendencia de Montevideo. [Centro de Gestión de Movilidad](https://montevideo.gub.uy/tipo/area-tematica/movilidad/gestion-de-la-movilidad/centro-de-gestion-de-movilidad).
- Intendencia de Montevideo. [Implementación del Proyecto Tránsito Adaptativo](https://montevideo.gub.uy/noticias/movilidad-y-transporte/implementacion-del-proyecto-transito-adaptativo). Publicado el 16 de agosto de 2024 y actualizado el 19 de agosto de 2024.
