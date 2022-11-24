import sys
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt
import numpy as np
# params = {'text.usetex': True, 'font.family': 'serif'}
# plt.rcParams.update(params)
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
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    pos = 0
    displayed = {
        'auc': 'Area under the ROC curve',
        'acc': 'Accuracy',
        'map': 'Mean average precision',
        'mean test variance': 'Mean variance'
    }
    permutation = [0, 1, 2, 3]
    for metric in sorted(set(data['metrics']['random']) - {
            'nb_train_samples', 'epoch', 'best epoch'}):
        if 'best' in metric or 'all' in metric or 'nll' in metric:
            continue
        # print(pos, metric)
        ax_i = 0
        ax_j = permutation[pos]
        axes[ax_j].title.set_text(displayed[metric])
        axes[ax_j].set_xlabel('Number of questions asked')
        for _, strategy in enumerate(['random', 'mean', 'variance']):
            if strategy in data['metrics']:
                x = np.array(data['metrics'][strategy]['nb_train_samples']) / 16
                print(strategy, metric, ' & '.join(map(str, np.round(data['metrics'][strategy][metric], 3))))
                axes[ax_j].plot(x, data['metrics'][strategy][metric], label=strategy, **get_style(strategy))
        plt.legend()
        pos += 1
    # for ip in range(2):
    #     for jp in range(3):
    #         axes[ip, jp].legend()
    fig_name = str(filename).replace('txt', 'after.pdf')
    plt.savefig(f'{fig_name}', bbox_inches='tight')
    return fig_name


if __name__ == '__main__':
    r = re.compile(r'-([0-9]+).txt')
    logfilename = sys.argv[1]

    with open(logfilename) as f:
        data = json.load(f)

    fig_name = plot_after(data, logfilename)
    os.system(f'open {fig_name}')
