"""Genera una demo HTML sin conexión con trazas calculadas por Python."""
import argparse
import json
from pathlib import Path
from src.experiment import ROOT, load_controller, run_episode
from src.data import load_windows, real_arrivals

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=['equilibrado','desigual','cambiante','estres','montevideo'], default='cambiante')
    p.add_argument('--seed', type=int, default=800000)
    p.add_argument('--train-seed', type=int, default=0)
    p.add_argument('--gif', action='store_true', help='Guardar también una animación de respaldo')
    args = p.parse_args()
    arrivals = None
    window_stamp = ''
    if args.scenario == 'montevideo':
        window_stamp, window = load_windows('test')[0]
        arrivals = real_arrivals(window, args.seed)
    payload = dict(scenario=args.scenario, seed=args.seed, train_seed=args.train_seed,
                   window=window_stamp, controllers=[])
    for name, label in [('fijo','Tiempos fijos'),('reglas','Mayor cola'),
                        ('q_learning','Q-learning'),('dqn','DQN')]:
        metrics, trace = run_episode(load_controller(name,args.train_seed),
                                    'equilibrado' if args.scenario=='montevideo' else args.scenario,
                                    args.seed, arrivals, record=True)
        payload['controllers'].append(dict(name=name,label=label,metrics=metrics,trace=trace))
    template = (ROOT/'assets'/'demo_template.html').read_text(encoding='utf-8')
    out = ROOT/'results'/'demo.html'
    out.write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False)),encoding='utf-8')
    print('Abrir en cualquier navegador:',out)
    if args.gif:
        make_gif(payload)

def make_gif(payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(2,2,figsize=(9,6))
    colors=['#89939f','#dc9851','#228d85','#355e9b']
    upper=max(max(r['ns'],r['eo']) for c in payload['controllers'] for r in c['trace'])+3
    def draw(t):
        for ax,c,color in zip(axes.flat,payload['controllers'],colors):
            ax.clear()
            r=c['trace'][t]
            ax.barh(['A','B'],[r['ns'],r['eo']],color=color)
            phase='Transición' if r['active']==-1 else ('Verde A' if r['active']==0 else 'Verde B')
            ax.set(xlim=(0,upper),title=c['label']+' · '+phase,xlabel='Vehículos esperando')
        fig.suptitle(f"Escenario {payload['scenario']} · segundo {t}")
        fig.tight_layout()
    animation=FuncAnimation(fig,draw,frames=range(0,600,5),interval=80)
    animation.save(ROOT/'results'/'demo.gif',writer=PillowWriter(fps=12))
    plt.close(fig)

if __name__ == '__main__':
    main()
