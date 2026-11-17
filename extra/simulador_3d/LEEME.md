# Semáforo RL en 3D

Visualización extra del proyecto: el cruce de Bv. Artigas y Rivera en 3D, con los
cuatro controladores (fijo, mayor cola, Q-learning y DQN) manejando el mismo tránsito.

**No es tránsito inventado.** Cada auto aparece en el segundo en que llegó y arranca en
el segundo en que cruzó según el simulador del proyecto (`src/entorno.py`). El marcador
y la tabla final salen de esos mismos datos. Se usa la primera semilla de tránsito de la
prueba oficial (800000) y la versión de los agentes entrenada con la semilla 0, así que
los números son de **un episodio**, no los promedios del informe.

## Cómo abrirlo

Abrir `index.html` con Chrome, Edge o Safari. Necesita internet la primera vez para
cargar Three.js y las tipografías.

Si el navegador no lo abre con doble clic, levantar un servidor local desde esta carpeta
(`extra/simulador_3d`) y entrar a <http://localhost:8000>:

```bash
python -m http.server 8000
```

## Controles

- **Escenarios:** los cuatro sintéticos y dos de Montevideo. "Conteo de 5 min" usa los datos
  del sensor multiplicados por 12; "unidad publicada" usa la lectura original, con el cruce casi vacío.
- **Marcador (derecha):** espera media por auto hasta ese segundo y autos en cola de cada
  controlador. Tocar un controlador lo muestra en la calle.
- **Cámaras:** aérea, a nivel de calle por Bv. Artigas o por Rivera, desde arriba y "Cine" (gira sola).
  Con el mouse se puede rotar, mover y hacer zoom.
- **Noche:** enciende faroles, ventanas y luces de los autos.
- **Barra de abajo:** reproducir o pausar (también con la barra espaciadora), adelantar o
  volver en el tiempo, y velocidad de 1× a 30×.

## Volver a generar los datos

`datos.js` se genera con el código del repo (`src/` y los modelos de `models/`).
Desde la raíz del repo:

```bash
python extra/simulador_3d/exportar_datos.py
```
