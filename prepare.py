from pathlib import Path
import argparse
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import pickle
import logging


def load_data(DATA):
    i = {}
    DATA_PATH = Path('data') / DATA
    df = pd.read_csv(DATA_PATH / 'data.csv')
    df['user'] = np.unique(df['user'], return_inverse=True)[1]  # Preprocess 0..N - 1
    df['item'] = np.unique(df['item'], return_inverse=True)[1]
    df['shifted_item'] = df['item'] + df['user'].nunique()
    i['trainval'] = pd.read_csv(DATA_PATH / 'trainval.csv')['index'].tolist()
    i['test'] = pd.read_csv(DATA_PATH / 'test.csv')['index'].tolist()
    df_trainval = df.loc[i['trainval'], ['user', 'shifted_item', 'rating']]
    df_test = df.loc[i['test'], ['user', 'shifted_item', 'rating']]
    X_train = df_trainval[['user', 'shifted_item']].to_numpy()
    y_train = df_trainval['rating'].to_numpy()
    X_test = df_test[['user', 'shifted_item']].to_numpy()
    y_test = df_test['rating'].to_numpy()
    return df['user'].nunique(), df['item'].nunique(), X_train, X_test, y_train, y_test

def prepare_data(DATA):
    i = {}
    DATA_PATH = Path('data') / DATA
    df = pd.read_csv(DATA_PATH / 'data.csv')
    df['user'] = np.unique(df['user'], return_inverse=True)[1]  # Preprocess 0..N - 1
    df['item'] = np.unique(df['item'], return_inverse=True)[1]
    df['shifted_item'] = df['item'] + df['user'].nunique()
    # print(df.nunique(), df.min(), df.max())
    print(df.agg(['min', 'max', 'nunique']))
    i['trainval'] = pd.read_csv(DATA_PATH / 'trainval.csv')['index'].tolist()
    logging.warning('min trainval index %d max index %d', min(i['trainval']), max(i['trainval']))
    i['test'] = pd.read_csv(DATA_PATH / 'test.csv')['index'].tolist()
    logging.warning('min test index %d max index %d', min(i['test']), max(i['test']))

    for dataset in ['trainval', 'test']:
        with open(DATA_PATH / f'{DATA}.{dataset}_libfm', 'w') as f:
            for user, item, rating in np.array(df.loc[i[dataset], ['user', 'shifted_item', 'rating']]):
                f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, item))


def prepare_ml_latest():
    df = pd.read_csv('ml-latest-small/ratings.csv')
    # df = df.query('movieId < 1000')
    print(df.head(), len(df))

    encode_user = dict(zip(df['userId'].unique(), range(10000)))
    encode_item = dict(zip(df['movieId'].unique(), range(10000)))
    encode_rating = lambda x: int(x >= 5)
    df['user_id'] = df['userId'].map(encode_user)
    df['item_id'] = df['movieId'].map(encode_item)
    df['value'] = df['rating'].map(encode_rating)

    print(df.head())

    ratings = coo_matrix((df['value'], (df['user_id'], df['item_id']))).tocsr()
    nb_users, nb_items = ratings.shape
    print('here', len(df), 'ratings', nb_users, 'users', nb_items, 'items')

    def convert(ratings):
        nb_users, _ = ratings.shape
        return list(filter(lambda x: len(x) > 1, [list(zip(ratings[i].indices, ratings[i].data)) for i in range(nb_users)]))

    print(ratings[-2:])

    data = convert(ratings)
    print(data[:3])
    print(data[-2:])
    print('here', len(df), 'ratings', len(data), 'users', nb_items, 'items')
    with open('ml0.pickle', 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(nb_items, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('data', type=str, nargs='?', default='movie100k')
    options = parser.parse_args()

    DATA = options.data

    if DATA == 'movie100k':
        prepare_data(DATA)
    elif DATA == 'ml-latest':
        prepare_ml_latest()
