"""Gráficos reproducibles a partir de los CSV de resultados."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.experiment import ROOT

NAMES = {'fijo':'Fijo', 'reglas':'Mayor cola', 'q_learning':'Q-learning', 'dqn':'DQN'}
COLORS = ['#89939f', '#dc9851', '#228d85', '#355e9b']

def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    out = ROOT / 'results' / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':150})
    summary = read_csv(ROOT/'results'/'summary.csv')
    scenarios = ['equilibrado','desigual','cambiante','estres','montevideo']
    algorithms = list(NAMES)
    for metric, title, ylabel in [
        ('mean_observed_wait','Espera observada por vehículo','Segundos por vehículo'),
        ('p95_observed_wait','Percentil 95 de espera observada','Segundos'),
        ('pending','Vehículos pendientes al finalizar','Vehículos')]:
        fig, ax = plt.subplots(figsize=(10,4.8))
        x = np.arange(len(scenarios))
        for j, algorithm in enumerate(algorithms):
            rows = [next(r for r in summary if r['scenario']==s and r['algorithm']==algorithm) for s in scenarios]
            values = [float(r[metric]) for r in rows]
            stds = [float(r[metric+'_std']) for r in rows]
            ax.bar(x+(j-1.5)*.19, values, width=.18, label=NAMES[algorithm],
                   color=COLORS[j], yerr=stds, capsize=2, error_kw={'lw':.7})
        ax.set_xticks(x, ['Equilibrado','Desigual','Cambiante','Estrés','Montevideo'])
        ax.set(ylabel=ylabel, title=title + ' (media y desvío)')
        ax.legend(ncol=4, frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out/(metric+'.png'))
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax, algorithm, color in zip(axes,['q_learning','dqn'],[COLORS[2],COLORS[3]]):
        curves = []
        for path in sorted((ROOT/'results').glob('training_'+algorithm+'_*.csv')):
            values = np.array([float(r['mean_observed_wait']) for r in read_csv(path)])
            smooth = np.convolve(values,np.ones(40)/40,mode='valid')
            ax.plot(np.arange(len(smooth))+40,smooth,color=color,alpha=.25,lw=.8)
            curves.append(smooth)
        if curves:
            mean = np.mean(curves,axis=0)
            ax.plot(np.arange(len(mean))+40,mean,color=color,lw=2.2)
        ax.set(title=NAMES[algorithm],xlabel='Episodio de entrenamiento')
    axes[0].set_ylabel('Espera observada (media móvil de 40 episodios)')
    fig.suptitle('Entrenamiento con exploración y mezcla de demandas')
    fig.tight_layout()
    fig.savefig(out/'learning.png')
    plt.close(fig)
    # Curvas por segundo en un episodio reservado, sin elegir el mejor agente.
    from src.experiment import load_controller,run_episode,write_csv
    fig, ax = plt.subplots(figsize=(10,4.4))
    trace_rows=[]
    for j,algorithm in enumerate(algorithms):
        _,trace=run_episode(load_controller(algorithm,0),'cambiante',800000,record=True)
        ax.plot([r['t'] for r in trace],[r['ns']+r['eo'] for r in trace],
                label=NAMES[algorithm],color=COLORS[j],alpha=.9)
        trace_rows.extend([dict(algorithm=algorithm,**r) for r in trace])
    ax.axvline(300,color='black',ls='--',lw=.8,label='Cambio de demanda')
    ax.set(xlabel='Segundo',ylabel='Vehículos esperando',title='Mismo tránsito y fase inicial para los cuatro controles')
    ax.legend(ncol=3,frameon=False,fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'queues_episode.png')
    plt.close(fig)
    write_csv(ROOT/'results'/'demo_trace.csv',trace_rows)
    print('Gráficos guardados en results/figures/')

if __name__ == '__main__':
    main()
