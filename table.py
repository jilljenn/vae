from collections import defaultdict
import pandas as pd
import json
import glob
import sys
import re


als = {
    'movie100k': (1.046, 439, 53),
    'movie1M': (1.635, 441, 524.596),
    'movie10M': (1.819, 407, 9830.521),
}

mcmc = {
    'movie100k': (0.991, 439, 62),
    # 'mangaki': 1.13,
    'movie1M': (0.938, 441, 5*60+25),
    'movie10M': (0.992, 407, 0),
    'fraction': (0.80, 500, 0)
}

hidden = {
    '1536127738',
    '1536125339',
    '1536132707',
    '1536148017',
    '1536123342',
    '1536123666'
}

results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for filename in glob.glob('oldresults/backup-*/*trainval*.txt'):
    if any(timestamp in filename for timestamp in hidden):
        continue

    print(filename)
    with open(filename) as f:
        data = json.load(f)

    m = re.search(r'backup-([^/]+)/', filename)
    category = m.group(1)
    if 'forced' in filename or 'nonsparse' in filename:
        category += '-forced'
        if data['args']['nb_batches'] != 10:
            continue

    if data['args']['sparse']:
        model = 'VFM+si'
    elif data['args']['degenerate']:
        model = 'MAP'
    else:
        model = 'VFM'
    timestamp = filename[-14:-4]
    # model += ' {:s}'.format(filename[-7:-4])

    sigma = data.get('sigma2')
    if sigma is None:
        m = re.search(r'trainval-([^-]*)-', filename)
        sigma = float(m.group(1))

    if 'forced' not in category:
        model += ' {:d}'.format(data['args']['nb_batches'])
    dataset = data['args']['data']
    metric = 'auc' if dataset == 'fraction' else 'rmse'

    if (dataset, metric) in results[category][model]:  # Duplicate
        print('Alert, duplicate for', model)
        print(timestamp)
        print(data['args'])
        model += ' ' + timestamp[-3:]

    results[category][model][(dataset, metric.upper())] = '\textbf{{{:.3f}}}'.format(data['metrics']['final ' + metric])
    results[category][model][(dataset, 'sigma')] = '{:.1f}'.format(sigma)
    if 'forced' not in category:
        results[category][model][(dataset, '\#batches')] = int(data['args']['nb_batches'])
        # results[category][model][(dataset, '$d$')] = int(data['args']['d'])
    # results[category][model][(dataset, 'batch time')] = '{:.3f}'.format(data['metrics']['time']['per_batch'])
    # results[category][model][(dataset, 'epoch time')] = '{:.3f}'.format(data['metrics']['time']['per_epoch'])
    results[category][model][(dataset, 'total time (s)')] = '{:.3f}'.format(data['metrics']['time']['total'])
    nb_epochs = int(str(data['stopped']).split('/')[0])
    results[category][model][(dataset, 'stopped at epoch')] = nb_epochs
    results[category][model][(dataset, 'time per epoch (s)')] = '{:.3f}'.format(data['metrics']['time']['total'] / nb_epochs)

for category in results:
    for dataset in mcmc:
        metric = 'auc' if dataset == 'fraction' else 'rmse'
        
        if ('5b' in category and dataset != 'fraction') or ('forced' in category and dataset == 'fraction'):
            continue

        rmse, nb_epochs, duration = mcmc[dataset]
        results[category]['FM MCMC'][dataset, metric.upper()] = '\textbf{{{:.3f}}}'.format(rmse)
        results[category]['FM MCMC'][dataset, 'stopped at epoch'] = nb_epochs
        if duration:
            results[category]['FM MCMC'][dataset, 'total time (s)'] = '{:.3f}'.format(duration)
            results[category]['FM MCMC'][dataset, 'time per epoch (s)'] = '{:.3f}'.format(duration / nb_epochs)
        if dataset in als:
            rmse, nb_epochs, duration = als[dataset]
            results[category]['FM ALS'][dataset, metric.upper()] = '\textbf{{{:.3f}}}'.format(rmse)
            results[category]['FM ALS'][dataset, 'stopped at epoch'] = nb_epochs
            if duration:
                results[category]['FM ALS'][dataset, 'total time (s)'] = '{:.3f}'.format(duration)
                results[category]['FM ALS'][dataset, 'time per epoch (s)'] = '{:.3f}'.format(duration / nb_epochs)
    df = pd.DataFrame.from_dict(results[category]).fillna('--')
    print(len(df.columns), 'columns')
    print(df.head())
    df[sorted(df.columns)].to_latex('/Users/jilljenn/code/article/aaai2019/table-{:s}.tex'.format(category), escape=False)
