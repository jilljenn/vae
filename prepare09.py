from scipy.sparse import coo_matrix
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle


df = pd.read_csv('skill_builder_data_processed.csv').sort_values('order_id')
print(df.head(), len(df))

encode_user = dict(zip(df['user_id'].unique(), range(10000)))
items = df['skill_id'].unique()
encode_item = dict(zip(items, range(10000)))
df['new_user_id'] = df['user_id'].map(encode_user)
df['item_id'] = df['skill_id'].map(encode_item)
print(df['correct'].unique())
print(df.head())

data = defaultdict(list)
for user_id, item_id, correct in np.array(df[['new_user_id', 'item_id', 'correct']]):
    data[user_id].append((item_id, correct))
data = list(filter(lambda x: len(x) > 1, list(data.values())))

print(data[:3])
print(data[-2:])
nb_items = len(items)
print('here', len(df), 'ratings', len(data), 'users', nb_items, 'items')
with open('assist09.pickle', 'wb') as f:
    pickle.dump(data, f)
    pickle.dump(nb_items, f)
