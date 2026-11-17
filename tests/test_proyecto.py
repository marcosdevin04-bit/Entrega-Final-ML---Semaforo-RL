"""Comprobaciones del simulador y de los algoritmos.

Uso:
    python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.entorno import Cruce, Configuracion, ESCENARIOS
from src.q_learning import QLearning, codificar
from src.datos import cargar_ventanas, llegadas_montevideo


class PruebasSimulador(unittest.TestCase):
    def test_no_se_pierden_autos_ni_esperas(self):
        """Todo auto que llegó cruzó o sigue en cola, y la espera total cierra."""
        for escenario in ESCENARIOS:
            cruce = Cruce()
            cruce.reiniciar(escenario, 42)
            termino = False
            while not termino:
                _, _, termino = cruce.avanzar(1)
            m = cruce.metricas()
            espera_pendientes = sum(cruce.t - llegada for cola in cruce.colas for llegada in cola)
            self.assertEqual(m['llegaron'], m['cruzaron'] + m['pendientes'])
            self.assertEqual(m['espera_total'], sum(cruce.esperas_cruzaron) + espera_pendientes)

    def test_transicion_y_verde_minimo(self):
        """No cambia antes de 10 s de verde, y durante la transición nadie cruza."""
        cruce = Cruce(Configuracion(duracion=40))
        llegadas = np.ones((40, 2), dtype=int)
        cruce.reiniciar(semilla=3, llegadas=llegadas)
        fase_inicial = cruce.fase
        cruce.avanzar(1)
        cruce.avanzar(1)
        self.assertEqual(cruce.fase, fase_inicial)
        antes = len(cruce.esperas_cruzaron)
        cruce.avanzar(1, registrar=True)
        self.assertEqual(cruce.fase, 1 - fase_inicial)
        self.assertEqual([r['fase_activa'] for r in cruce.historial[:3]], [-1, -1, -1])
        self.assertEqual(len(cruce.esperas_cruzaron) - antes, 2)

    def test_verde_maximo(self):
        """Aunque el controlador nunca pida cambiar, el verde no pasa de 60 s."""
        cruce = Cruce()
        cruce.reiniciar(semilla=1)
        termino = False
        while not termino:
            _, _, termino = cruce.avanzar(0)
            self.assertLessEqual(cruce.edad_verde, cruce.config.verde_maximo)
        self.assertGreater(cruce.cambios, 0)

    def test_llegadas_no_dependen_de_las_acciones(self):
        a, b = Cruce(), Cruce()
        a.reiniciar('cambiante', 8)
        b.reiniciar('cambiante', 8)
        for _ in range(10):
            a.avanzar(0)
            b.avanzar(1)
        np.testing.assert_array_equal(a.llegadas, b.llegadas)

    def test_sin_llegadas_y_ultimo_paso_incompleto(self):
        """Con 13 s de duración el último paso dura 3 s; sin autos no hay espera."""
        cruce = Cruce(Configuracion(duracion=13))
        cruce.reiniciar(llegadas=np.zeros((13, 2)))
        termino, recompensa_total = False, 0
        while not termino:
            _, recompensa, termino = cruce.avanzar(0)
            recompensa_total += recompensa
        self.assertEqual(cruce.t, 13)
        self.assertEqual(recompensa_total, 0)
        self.assertEqual(cruce.metricas()['espera_media'], 0)

    def test_particion_por_fechas(self):
        for particion, desde, hasta in [('entrenamiento', 1, 20), ('validacion', 21, 25), ('prueba', 26, 31)]:
            for momento, tasas in cargar_ventanas(particion):
                self.assertTrue(desde <= int(momento[8:10]) <= hasta)
                self.assertTrue(np.all(tasas >= 0))
            llegadas = llegadas_montevideo(cargar_ventanas(particion)[0][1], 42)
            self.assertEqual(llegadas.shape, (600, 2))


class PruebasAprendizaje(unittest.TestCase):
    def test_q_learning_actualiza_al_terminar_y_guarda(self):
        agente = QLearning(alpha=1)
        obs = np.zeros(5, dtype=np.float32)
        agente.q[:] = 100
        agente.actualizar(obs, 1, -3, obs, True)
        self.assertEqual(agente.q[codificar(obs) + (1,)], -3)   # al terminar, Q = r
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'q.npz'
            agente.guardar(ruta)
            np.testing.assert_array_equal(agente.q, QLearning.cargar(ruta).q)

    def test_dqn_entrena_sin_tocar_la_red_objetivo_y_guarda(self):
        import torch
        from src.dqn import DQN
        agente = DQN(semilla=0)
        obs = np.ones(5, dtype=np.float32)
        for _ in range(1000):
            agente.recordar(obs, 0, -1, obs, True)
        objetivo_antes = [p.detach().clone() for p in agente.red_objetivo.parameters()]
        red_antes = [p.detach().clone() for p in agente.red.parameters()]
        perdida = agente.aprender()
        self.assertTrue(np.isfinite(perdida))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(red_antes, agente.red.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(objetivo_antes, agente.red_objetivo.parameters())))
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'dqn.pt'
            agente.guardar(ruta)
            restaurado = DQN.cargar(ruta)
            with torch.no_grad():
                np.testing.assert_allclose(agente.red(torch.from_numpy(obs)),
                                           restaurado.red(torch.from_numpy(obs)))


if __name__ == '__main__':
    unittest.main()
