import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import sys
import os


log_name = sys.argv[1]
K = 5  # Strip length
fig_name = log_name.replace('txt', 'pdf')
print(log_name)
cv = 'trainval' not in log_name

als = {
    'movie100k': 'rlog-data-movie100k-X_fm-1536156951.csv',
    'movie1M': 'rlog-data-movie1M-X_fm-1536158156.csv'
}

mcmc = {
    # 'movie100k': 'rlog-data-movie100k-X_fm-1536112221.csv',
    'movie100k': 'rlog-data-movie100k-X_fm-1536157242.csv',
    'movie1M': 'rlog-data-movie1M-X_fm-1536142042.csv',
    'movie10M': 'rlog-data-movie10M-X_fm-1536149311.csv'
}

criteria = defaultdict(lambda: defaultdict(list))
with open(log_name) as f:
    data = json.load(f)

dataset = data['args']['data']
metric_name = 'auc' if dataset == 'fraction' else 'rmse'

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

if metric_name in data['metrics']['train']:
    metric.plot(train_epochs, data['metrics']['train'][metric_name], label='train {:s}'.format(metric_name))
metric.plot(data['metrics']['test']['epoch'], data['metrics']['test'][metric_name], label='VFM')
MAX_EPOCH = max(data['metrics']['test']['epoch'])

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
        df = pd.read_csv(mcmc[dataset])
        metric.plot(1 + df.index[:MAX_EPOCH], df['rmse'][:MAX_EPOCH], label='libFM MCMC')

    if dataset in als:
        df = pd.read_csv(als[dataset])
        metric.plot(1 + df.index[:MAX_EPOCH], df['rmse'][:MAX_EPOCH], label='libFM ALS')

metric.set_title('Test {:s} ↓ over epochs'.format(metric_name.upper()))
metric.legend()
if metric_name == 'rmse':
    metric.set_ylim(ymax=2)
# fig.legend()
fig.savefig('{:s}'.format(fig_name, format='pdf'))
os.system('open {:s}'.format(fig_name))
