import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import os
# params = {'text.usetex': True, 'font.family': 'serif'}
# plt.rcParams.update(params)
default_cycler = list(plt.rcParams['axes.prop_cycle'])


def get_style_and_label(label):
    for pos, key in enumerate(['VFM', 'MCMC', 'VBFM', 'OVBFM']):
        if key in label:
            style = default_cycler[pos].copy()
            if 'mean' in label:
                style['lw'] = 2.5
                style['linestyle'] = 'dashed'
            elif 'valid' in label:
                style['linestyle'] = 'dotted'
    '''if 'best' in label:
                    style['linestyle'] = '--'''
    style['label'] = label
    return style


log_name = sys.argv[1]
K = 5  # Strip length
fig_name = log_name.replace('txt', 'pdf')
fig_name_png = log_name.replace('txt', 'png')
print(log_name)
cv = 'trainval' not in log_name and 'ongoing_test' not in log_name
cv = False

LIBFM_RESULTS_PATH = Path('../Scalable-Variational-Bayesian-Factorization-Machine/results/')
MAX_EPOCH = 200 # max(data['metrics']['test']['epoch'])

criteria = defaultdict(lambda: defaultdict(list))
with open(log_name) as f:
    data = json.load(f)
embedding_size = data['args']['d']
dataset = data['args']['data']
metric_name = 'acc' if dataset in {
    'fraction', 'movie5', 'movie20', 'movie100',
    'movie100k-binary', 'movie1M-binary'} else 'rmse'
CPP_METRIC = {
    'last': f'{metric_name}_mcmc_this',
    'mean': f'{metric_name}_mcmc_all'
}
METHODS = {
    'vb': ('VBFM', ['last']),
    'vb_online': ('OVBFM', ['last']),
    'mcmc': ('MCMC', ['last', 'mean'])
}

train_epochs = np.unique(sorted(data['metrics']['train']['epoch']))
test_epochs = data['metrics']['test']['epoch'][:MAX_EPOCH]
print('Train', min(train_epochs), max(train_epochs))
train = data['metrics']['train']['elbo']

# Track progress on train
for i, v in enumerate(train[K - 1:], start=K - 1):
    epoch = train_epochs[i]
    strip = train[i - K + 1:i + 1]

    if max(strip) == 0:
        progress = 0.
    else:
        progress = 1000 * (sum(strip) / (K * max(strip)) - 1)

    criteria['progress']['epoch'].append(epoch)
    criteria['progress']['value'].append(progress)

all_progress = dict(zip(criteria['progress']['epoch'], criteria['progress']['value']))

# Track generalization loss on valid
if cv:
    valid = data['metrics']['valid'][metric_name]

    for i, v in enumerate(valid):
        epoch = data['metrics']['valid']['epoch'][i]

        gen_loss = 100 * (v / min(valid[:i + 1]) - 1)
        if all_progress[epoch] == 0:
            quotient = 0.
        else:
            quotient = gen_loss / all_progress[epoch]

        criteria['gen_loss']['epoch'].append(epoch)
        criteria['gen_loss']['value'].append(gen_loss)
        criteria['quotient']['epoch'].append(epoch)
        criteria['quotient']['value'].append(quotient)


if cv:
    fig, ((elbo, metric), (progress_graph, criterion_graph)) = plt.subplots(2, 2, figsize=(8, 8))
else:
    fig, metric = plt.subplots(1, 1, figsize=(4, 4))
    order = '↓' if metric_name == 'rmse' else '↑'
    metric.set_title(f'Test {metric_name.upper()} {order} over epochs')
    metric.set_xlabel('Epochs')
    metric.set_ylabel(metric_name.upper())

if metric_name in data['metrics']['train']:
    metric.plot(train_epochs, data['metrics']['train'][metric_name], label='train {:s}'.format(metric_name))
metric.plot(test_epochs, data['metrics']['test'][metric_name][:MAX_EPOCH], **get_style_and_label('VFM last'))
if 'rmse_all_of_mean' in data['metrics']['test']:
    metric.plot(test_epochs, data['metrics']['test']['rmse_all_of_mean'][:MAX_EPOCH], **get_style_and_label('VFM mean'))
    print('VFM mean', data['metrics']['test']['rmse_all_of_mean'][:MAX_EPOCH][-1])
elif 'acc_all' in data['metrics']['test']:
    metric.plot(test_epochs, data['metrics']['test']['acc_all'], **get_style_and_label('VFM mean'))
    print('VFM mean ACC', data['metrics']['test']['acc_all'][:MAX_EPOCH][-1])
    print('VFM mean AUC', data['metrics']['test']['auc_all'][:MAX_EPOCH][-1])
    print('VFM mean MAP', data['metrics']['test']['map_all'][:MAX_EPOCH][-1])


if cv:
    elbo.plot(train_epochs, data['metrics']['train']['elbo'], label='train elbo')
    elbo.set_title('Elbo ↑ over epochs')

    metric.plot(data['metrics']['valid']['epoch'], data['metrics']['valid'][metric_name], label='valid')
    criterion_graph.hlines(0.2, min(criteria['quotient']['epoch']), max(criteria['quotient']['epoch']))

    for criterion in {'gen_loss', 'quotient'}:# if cv else {'progress'}:
        criterion_graph.plot(criteria[criterion]['epoch'], criteria[criterion]['value'], label=criterion)
    progress_graph.plot(criteria['progress']['epoch'], criteria['progress']['value'], label='progress')

    criterion_graph.set_title('Stopping rules over epochs')
    criterion_graph.legend()
    progress_graph.set_title('Progress of ELBO')


for method, (displayed_method, metrics) in METHODS.items():
    log_path = LIBFM_RESULTS_PATH / f'{method}_{dataset}_{embedding_size}.csv'
    if log_path.is_file():
        df = pd.read_csv(log_path, sep='\t')
        if len(df) == 0:
            continue
        for eval_type in metrics:
            metric.plot(1 + df.index[:MAX_EPOCH],
                df[CPP_METRIC[eval_type]][:MAX_EPOCH],
                **get_style_and_label(f'{displayed_method} {eval_type}'))
            print(f'{displayed_method} {eval_type}', df[CPP_METRIC[eval_type]][:MAX_EPOCH].tail(1))


metric.legend()
if metric_name == 'rmse':
    metric.set_ylim(ymax=1.2)
# fig.legend()
plt.tight_layout()
fig.savefig('{:s}'.format(fig_name_png, format='png', bbox_inches='tight'))
fig.savefig('{:s}'.format(fig_name, format='pdf'))#, bbox_inches='tight'))
os.system('open {:s}'.format(fig_name))
# plt.show()
