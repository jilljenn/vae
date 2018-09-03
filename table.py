from collections import defaultdict
import pandas as pd
import json
import glob
import re


mcmc = {
    'movie100k': 0.991,
    'mangaki': 1.13,
    'movie1M': 0.938,
    'movie10M': 0.992
}

results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for filename in glob.glob('old results/*/*trainval*'):
    with open(filename) as f:
        data = json.load(f)

    m = re.search(r'backup-([^/]+)/', filename)
    category = m.group(1)

    if data['args']['sparse']:
        model = 'VFM sparse'
    elif data['args']['degenerate']:
        model = 'FM MAP'
    else:
        model = 'VFM'

    sigma = data.get('sigma')
    if sigma is None:
        m = re.search(r'trainval-(.*)-', filename)
        sigma = float(m.group(1))

    dataset = data['args']['data']

    results[category][model][(dataset, 'rmse')] = data['metrics']['final rmse']
    results[category][model][(dataset, 'sigma')] = sigma
    results[category][model][(dataset, 'batch time')] = data['metrics']['time']['per_batch']
    results[category][model][(dataset, 'epoch time')] = data['metrics']['time']['per_epoch']
    results[category][model][(dataset, 'total time')] = data['metrics']['time']['total']
    results[category][model][(dataset, 'stopped')] = data['stopped']

for category in results:
    for dataset in mcmc:
        results[category]['FM MCMC'][dataset, 'rmse'] = mcmc[dataset]
    df = pd.DataFrame.from_dict(results[category]).round(3).fillna('--')
    print(df.head())
    df.to_latex('/Users/jilljenn/code/article/aaai2019/table-{:s}.tex'.format(category))
