"""Funciones compartidas por los scripts de entrenamiento, evaluación y demo."""
from pathlib import Path
import json
from .entorno import Cruce
from .q_learning import QLearning
from .controladores import SemaforoFijo, MayorCola

RAIZ = Path(__file__).resolve().parents[1]
CONTROLADORES = ('fijo', 'mayor_cola', 'q_learning', 'dqn')
NOMBRES = {'fijo': 'Fijo', 'mayor_cola': 'Mayor cola', 'q_learning': 'Q-learning', 'dqn': 'DQN'}


def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding='utf-8'))


def cargar_controlador(nombre, semilla=0):
    """Devuelve el controlador pedido. Los agentes se cargan de models/ según su semilla."""
    if nombre == 'fijo':
        verde = leer_json(RAIZ / 'results' / 'seleccion_fijo.json')['verde_elegido']
        return SemaforoFijo(verde)
    if nombre == 'mayor_cola':
        return MayorCola()
    if nombre == 'q_learning':
        return QLearning.cargar(RAIZ / 'models' / f'q_learning_{semilla}.npz')
    if nombre == 'dqn':
        from .dqn import DQN   # se importa acá para no exigir PyTorch si no se usa DQN
        return DQN.cargar(RAIZ / 'models' / f'dqn_{semilla}.pt')
    raise ValueError('Controlador desconocido: ' + nombre)


def correr_episodio(controlador, escenario, semilla, llegadas=None, registrar=False):
    """Simula un episodio completo sin exploración y devuelve (métricas, historial)."""
    cruce = Cruce()
    obs = cruce.reiniciar(escenario, semilla, llegadas)
    termino = False
    while not termino:
        obs, _, termino = cruce.avanzar(controlador.actuar(obs), registrar=registrar)
    return cruce.metricas(), cruce.historial
