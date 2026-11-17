# Decisiones de diseño

Documento de decisiones del proyecto, elaborado el 12 de noviembre de 2026.

- **Una intersección y dos accesos.** Permite concentrarse en el efecto de las
  políticas y entender el simulador. No reproduce una red vial.
- **Q-learning y DQN.** Permiten comparar una tabla discretizada con aproximación
  neuronal sobre las mismas variables y presupuesto de interacción.
- **Implementación pequeña propia.** Facilita leer las reglas y las actualizaciones
  en archivos Python, sin depender de una biblioteca de entornos.
- **Reglas como comparación fuerte.** Mayor cola permite comprobar si entrenar
  un agente aporta algo frente a una solución de bajo costo.
- **Fijo elegido en validación.** Se comparan ocho duraciones con semillas reservadas,
  usando una duración común para ambos accesos. No representa un plan vial óptimo.
- **Demanda pregenerada.** Las acciones nunca modifican el generador aleatorio
  de llegadas, lo que permite comparaciones pareadas.
- **Datos de Montevideo.** Se conserva una selección literal y se documenta toda
  transformación. Los resultados principales usan la unidad publicada (veh/h).
- **Unidad de `volumen_hora`.** Después de la prueba se encontró evidencia de que el
  archivo trae conteos de cinco minutos. Se agregó como análisis de sensibilidad
  (`scripts/sensibilidad_montevideo.py`), con los mismos modelos y sin reentrenar,
  en vez de reemplazar los resultados principales después de ver la prueba.
- **Resultados locales negativos.** Se conservan los resultados en los que los
  agentes pierden. No se vuelven a ajustar hiperparámetros usando el conjunto de prueba.
