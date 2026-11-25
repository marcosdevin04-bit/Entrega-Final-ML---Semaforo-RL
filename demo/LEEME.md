# Semáforo RL en 3D

Demo del proyecto: el cruce de Bv. Artigas y Rivera en 3D, con los
cuatro controladores (fijo, mayor cola, Q-learning y DQN) manejando el mismo tránsito.

**No es tránsito inventado.** Cada auto aparece en el segundo en que llegó y arranca en
el segundo en que cruzó según el simulador del proyecto (`src/entorno.py`). El marcador
y la tabla final salen de esos mismos datos. Se usa la primera semilla de tránsito de la
prueba oficial (800000) y la versión de los agentes entrenada con la semilla 0, así que
los números son de **un episodio**, no los promedios del informe.

## Cómo abrirlo

Desde la raíz del repo:

```bash
python -m scripts.demo
```

Genera los episodios, levanta un servidor local y abre el simulador en el navegador.
Para cerrarlo, `Ctrl+C` en la terminal. Necesita internet para cargar Three.js y las tipografías.

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
Para regenerarlo sin abrir el navegador:

```bash
python -m scripts.demo --solo-datos
```
