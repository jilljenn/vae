from collections import defaultdict
import pandas as pd
import json
import glob
import re


results = defaultdict(lambda: defaultdict(dict))
for filename in glob.glob('results/*trainval*'):
    with open(filename) as f:
        data = json.load(f)

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

    results[model][(dataset, 'rmse')] = data['metrics']['final rmse']
    results[model][(dataset, 'sigma')] = sigma
    results[model][(dataset, 'batch time')] = data['metrics']['time']['per_batch']
    results[model][(dataset, 'epoch time')] = data['metrics']['time']['per_epoch']
    results[model][(dataset, 'total time')] = data['metrics']['time']['total']
    results[model][(dataset, 'stopped')] = data['stopped']

df = pd.DataFrame.from_dict(results)
print(df.head())
df.to_latex('/Users/jilljenn/code/article/aaai2019/table.tex')
