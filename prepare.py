from scipy.sparse import coo_matrix
import pandas as pd
import pickle


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
