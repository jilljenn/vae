import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
import sys


log_name = sys.argv[1]
K = 5  # Strip length
fig_name = log_name.replace('txt', 'pdf')
print(log_name)
cv = 'trainval' not in log_name

criteria = defaultdict(lambda: defaultdict(list))
with open(log_name) as f:
    data = json.load(f)

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
    valid = data['metrics']['valid']['rmse']

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


fig, ((elbo, rmse), (_, criterion_graph)) = plt.subplots(2, 2, figsize=(8, 8))
elbo.plot(train_epochs, data['metrics']['train']['elbo'], label='train elbo')
if 'rmse' in data['metrics']['train']:
    rmse.plot(train_epochs, data['metrics']['train']['rmse'], label='train rmse')
rmse.plot(data['metrics']['test']['epoch'], data['metrics']['test']['rmse'], label='test')

if cv:
    rmse.plot(data['metrics']['valid']['epoch'], data['metrics']['valid']['rmse'], label='valid')
    criterion_graph.hlines(0.2, min(criteria['quotient']['epoch']), max(criteria['quotient']['epoch']))

for criterion in {'gen_loss', 'quotient'} if cv else {'progress'}:
    criterion_graph.plot(criteria[criterion]['epoch'], criteria[criterion]['value'], label=criterion)

rmse.set_title('RMSE ↓ over epochs')
rmse.set_ylim(ymax=2)
elbo.set_title('Elbo ↑ over epochs')
criterion_graph.set_title('Stopping rules over epochs')
fig.legend()
fig.savefig('{:s}'.format(fig_name, format='pdf'))
