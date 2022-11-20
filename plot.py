import sys
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt
params = {'text.usetex': True, 'font.family': 'serif'}
plt.rcParams.update(params)
default_cycler = list(plt.rcParams['axes.prop_cycle'])
import os


def get_style(label):
    style = {}  # default_cycler[pos].copy()
    """for pos, key in enumerate(['acc', 'auc', 'map']):
                    if key in label:
                        style = default_cycler[pos].copy()
                        style['lw'] = 1 + pos
                if 'best' in label:
                    style['linestyle'] = '--'"""
    return style


def plot_after(data, filename):
    plt.clf()
    fig, axes = plt.subplots(4, 4, figsize=(21, 7), sharey='row')
    for pos, metric in enumerate(set(data['metrics']['random']) - {
            'nb_train_samples', 'epoch', 'best epoch'}):
        print(pos, metric)
        ax_i = pos // 4
        ax_j = pos % 4
        axes[ax_i, ax_j].title.set_text(metric)
        for _, strategy in enumerate(['random', 'mean', 'variance']):
            if strategy in data['metrics']:
                x = data['metrics'][strategy]['nb_train_samples']
                axes[ax_i, ax_j].plot(x, data['metrics'][strategy][metric], label=strategy, **get_style(strategy))
        axes[ax_i, ax_j].legend()
    # for ip in range(2):
    #     for jp in range(3):
    #         axes[ip, jp].legend()
    fig_name = str(filename).replace('txt', 'after.pdf')
    plt.savefig(f'{fig_name}')
    return fig_name


if __name__ == '__main__':
    r = re.compile(r'-([0-9]+).txt')
    logfilename = sys.argv[1]

    with open(logfilename) as f:
        data = json.load(f)

    fig_name = plot_after(data, logfilename)
    os.system(f'open {fig_name}')
