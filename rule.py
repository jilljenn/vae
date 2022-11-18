import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import os


log_name = sys.argv[1]
K = 5  # Strip length
fig_name = log_name.replace('txt', 'pdf')
fig_name_png = log_name.replace('txt', 'png')
print(log_name)
cv = 'trainval' not in log_name

LIBFM_RESULTS_PATH = Path('../Scalable-Variational-Bayesian-Factorization-Machine/results/')

als = {
}

vbfm = {
    'movie100k': LIBFM_RESULTS_PATH / 'vbfm_100k.csv',
    'movie100k-binary': LIBFM_RESULTS_PATH / 'vbfm_100k_binary.csv',
}

ovbfm = {
    'movie100k': LIBFM_RESULTS_PATH / 'ovbfm_100k.csv',
    'movie1M': LIBFM_RESULTS_PATH / 'ovbfm_1M',
    'movie10M': LIBFM_RESULTS_PATH / 'ovbfm_10M'
}

mcmc = {
    'movie100': LIBFM_RESULTS_PATH / 'mcmc_100',
    'movie1000': LIBFM_RESULTS_PATH / 'mcmc_1000',
    'movie100k': LIBFM_RESULTS_PATH / 'mcmc_100k.csv',
    'movie100k-binary': LIBFM_RESULTS_PATH / 'mcmc_100k_binary.csv',
    'movie1M': LIBFM_RESULTS_PATH / 'mcmc_1M',
    'movie1M-binary': LIBFM_RESULTS_PATH / 'mcmc_1M_binary.csv',
    'movie10M': LIBFM_RESULTS_PATH / 'mcmc_10M'
}

criteria = defaultdict(lambda: defaultdict(list))
with open(log_name) as f:
    data = json.load(f)

dataset = data['args']['data']
metric_name = 'acc' if dataset in {
    'fraction', 'movie5', 'movie20', 'movie100',
    'movie100k-binary', 'movie1M-binary'} else 'rmse'
CPP_METRIC = f'{metric_name}_mcmc_this'

train_epochs = np.unique(sorted(data['metrics']['train']['epoch']))
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
    metric.set_title('Test {:s} ↓ over epochs'.format(metric_name.upper()))
    metric.set_xlabel('Epochs')
    metric.set_ylabel(metric_name.upper())

if metric_name in data['metrics']['train']:
    metric.plot(train_epochs, data['metrics']['train'][metric_name], label='train {:s}'.format(metric_name))
metric.plot(data['metrics']['test']['epoch'], data['metrics']['test'][metric_name], label='VFM this')
if 'rmse_all' in data['metrics']['test']:
    metric.plot(data['metrics']['test']['epoch'], data['metrics']['test']['rmse_all'], label='VFM all')
elif 'acc_all' in data['metrics']['test']:
    metric.plot(data['metrics']['test']['epoch'], data['metrics']['test']['acc_all'], label='VFM all')

# print('VFM', data['metrics']['test'][metric_name])
MAX_EPOCH = 400 # max(data['metrics']['test']['epoch'])
# 200 # 

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
else:
    if dataset in mcmc:
        df = pd.read_csv(mcmc[dataset], sep='\t')
        print(df.columns)
        mcmc_metric_names = (
            ['acc_mcmc_this', 'acc_mcmc_all'] if metric_name == 'acc'
            else ['rmse_mcmc_this', 'rmse_mcmc_all'])
        for mcmc_metric_name in mcmc_metric_names:
            metric.plot(1 + df.index[:MAX_EPOCH],
                df[mcmc_metric_name][:MAX_EPOCH],
                label=f'libFM MCMC {mcmc_metric_name}')
        # print('MCMC', df[mcmc_metric_name][:MAX_EPOCH])

    if dataset in vbfm:
        df = pd.read_csv(vbfm[dataset], sep='\t')
        metric.plot(1 + df.index[:MAX_EPOCH], df[CPP_METRIC][:MAX_EPOCH], label='VBFM this')

    if dataset in ovbfm:
        df = pd.read_csv(ovbfm[dataset], sep='\t')
        metric.plot(1 + df.index[:MAX_EPOCH], df[CPP_METRIC][:MAX_EPOCH], label='OVBFM this')
        # print('OVBFM', df['rmse_mcmc_all'][:MAX_EPOCH])

    if dataset in als:
        df = pd.read_csv(als[dataset])
        metric.plot(1 + df.index[:MAX_EPOCH], df['rmse'][:MAX_EPOCH], label='libFM ALS')

metric.legend()
if metric_name == 'rmse':
    metric.set_ylim(ymax=1.2)
# fig.legend()
fig.savefig('{:s}'.format(fig_name_png, format='png', bbox_inches='tight'))
fig.savefig('{:s}'.format(fig_name, format='pdf', bbox_inches='tight'))
os.system('open {:s}'.format(fig_name))
