"""Q-learning tabular.

La tabla Q guarda, para cada situación (estado) y cada acción, cuánta
recompensa futura se espera. Como la observación tiene números continuos,
primero se agrupan en intervalos para que la cantidad de estados sea finita.
"""
import numpy as np

# Bordes de los intervalos. Ejemplo con las colas:
# 0 autos -> grupo 0, 1-2 -> 1, 3-5 -> 2, 6-10 -> 3, 11-20 -> 4, 21-40 -> 5, más de 40 -> 6
BORDES_COLA = np.array([0, 2, 5, 10, 20, 40])
# Edad del verde: 0-9 s -> 0, 10-19 s -> 1, ..., 50-59 s -> 5, 60 s -> 6
BORDES_EDAD = np.array([9, 19, 29, 39, 49, 59])
# Tamaño de la tabla: cola A (7) x cola B (7) x fase (2) x edad (7) x tiempo (3) x acción (2)
FORMA_TABLA = (7, 7, 2, 7, 3, 2)


def codificar(obs):
    """Convierte la observación en el índice de una fila de la tabla (el estado)."""
    cola_a = round(float(obs[0]) * 40)        # se deshace la escala /40
    cola_b = round(float(obs[1]) * 40)
    fase = int(obs[2])
    edad = round(float(obs[3]) * 60)          # se deshace la escala /60
    grupo_a = int(np.digitize(cola_a, BORDES_COLA, right=True))
    grupo_b = int(np.digitize(cola_b, BORDES_COLA, right=True))
    grupo_edad = int(np.digitize(edad, BORDES_EDAD, right=True))
    tercio = min(2, int(float(obs[4]) * 3))   # principio, mitad o final del episodio
    return (grupo_a, grupo_b, fase, grupo_edad, tercio)


class QLearning:
    def __init__(self, semilla=0, alpha=.12, gamma=.97):
        self.q = np.zeros(FORMA_TABLA, dtype=np.float32)  # la tabla empieza en cero
        self.rng = np.random.default_rng(semilla)
        self.alpha = alpha   # tasa de aprendizaje: cuánto se corrige cada vez
        self.gamma = gamma   # descuento: cuánto pesan las recompensas futuras

    def actuar(self, obs, epsilon=0.):
        """Epsilon-greedy: con probabilidad epsilon explora al azar; si no, elige la mejor acción."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(2))
        valores = self.q[codificar(obs)]   # [valor de mantener, valor de cambiar]
        return int(np.argmax(valores))

    def actualizar(self, obs, accion, recompensa, siguiente, termino):
        """Regla de Q-learning: Q(s,a) <- Q(s,a) + alpha * (objetivo - Q(s,a))."""
        celda = codificar(obs) + (accion,)
        if termino:
            objetivo = recompensa   # el episodio terminó: no hay futuro que sumar
        else:
            mejor_siguiente = float(self.q[codificar(siguiente)].max())
            objetivo = recompensa + self.gamma * mejor_siguiente
        error = objetivo - self.q[celda]
        self.q[celda] += self.alpha * error

    def guardar(self, ruta):
        np.savez_compressed(ruta, q=self.q)

    @classmethod
    def cargar(cls, ruta):
        agente = cls()
        with np.load(ruta, allow_pickle=False) as datos:
            agente.q = datos['q'].copy()
        return agente
