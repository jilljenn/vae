from itertools import product
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, average_precision_score
from scipy.sparse import coo_matrix, load_npz, save_npz, hstack, find
from scipy.stats import sem, t
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
from collections import defaultdict
from datetime import datetime
import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow_probability as tfp
from pathlib import Path
wtfd = tfp.distributions
import argparse
import os
from plot import plot_after
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from pathlib import Path
import os.path
import getpass
import logging
import pandas as pd
import numpy as np
import random
import pickle
import yaml
import json
import time
from prepare import load_data


DESCRIPTION = 'Rescaled mode and test every step'
SUFFIX = 'forced'
start_time = time.time()

parser = argparse.ArgumentParser(description='Run VFM')
parser.add_argument('data', type=str, nargs='?', default='fraction')
parser.add_argument('--degenerate', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--sparse', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--regression', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--classification', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--load', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--single_user', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--split_valid', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--interactive', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--valid_patience', type=int, nargs='?', default=10)
parser.add_argument('--train_patience', type=int, nargs='?', default=4)
parser.add_argument('--d', type=int, nargs='?', default=3)
parser.add_argument('--lr', type=float, nargs='?', default=0.01)
parser.add_argument('--nb_batches', type=int, nargs='?', default=1)
parser.add_argument('--min_epochs', type=int, nargs='?', default=200)
parser.add_argument('--max_epochs', type=int, nargs='?', default=200)
parser.add_argument('--var_samples', type=int, nargs='?', default=1)
parser.add_argument('--method', type=str, nargs='?', default='adam')
parser.add_argument('--link', type=str, nargs='?', default='softplus')

parser.add_argument('--v', type=int, nargs='?', default=1) # Verbose
options = parser.parse_args()


# Config
DATA = options.data
DATA_PATH = Path('data') / DATA
logging.warning('Data is %s', DATA)
VERBOSE = options.v
NB_VARIATIONAL_SAMPLES = options.var_samples
COMPUTE_TEST_EVERY = 1
N_QUESTIONS_ASKED = 20
TRAIN_EVERY_N_QUESTIONS = 4
MIN_EPOCHS = options.min_epochs
MAX_EPOCHS = options.max_epochs
embedding_size = options.d
nb_iters = options.nb_batches
learning_rate = options.lr  # learning_rate 0.001 works better for classification
if options.classification:
    learning_rate = 0.1
link = tf.nn.softplus if options.link == 'softplus' else tf.math.abs


# Load data
if DATA in {'fraction', 'movie1M', 'movie10M', 'movie100k',
    'movie100k-binary', 'movie1M-binary'}:
    df = pd.read_csv(DATA_PATH / 'data.csv')
    print('Starts at', df['user'].min(), df['item'].min())
    try:
        with open(DATA_PATH / 'config.yml') as f:
            config = yaml.load(f)
            nb_users = config['nb_users']
            nb_items = config['nb_items']
    except IOError:  # Guess from data
        nb_users = 1 + df['user'].max()
        nb_items = 1 + df['item'].max()
    df['item'] += nb_users
    print(df.head())
else:
    df = pd.read_parquet(DATA_PATH / 'data.parquet', engine='fastparquet').reset_index(drop=True)
    # print(len(df))
    missing = pd.DataFrame(list(set(product(df['userId'].unique(), df['movieId'].unique())) -
                                set(list(map(tuple, df[['userId', 'movieId']].values.tolist())))), columns=df.columns)
    missing['outcome'] = 0
    df['outcome'] = 1
    # print(len(df) / (df['userId'].nunique() * df['movieId'].nunique()))
    df = pd.concat((df, missing)).reset_index(drop=True)
    # print(df.head())
    df['user'] = np.unique(df['userId'], return_inverse=True)[1]
    df['item'] = np.unique(df['movieId'], return_inverse=True)[1]
    nb_users = 1 + df['user'].max()
    nb_items = 1 + df['item'].max()
    df['item'] += nb_users
    print(df)
    # sys.exit(0)

# Is it classification or regression?
if options.regression or 'rating' in df:
    is_regression = True
    is_classification = False
if DATA.endswith('binary') or (options.classification and 'outcome' in df):
    is_classification = True
    is_regression = False

nb_entries = len(df)

# Build sparse features
X_fm_file = str(DATA_PATH / 'X_fm.npz')
if True: # not os.path.isfile(X_fm_file):
    rows = np.arange(nb_entries).repeat(2)
    cols = np.array(df[['user', 'item']]).flatten()
    data = np.ones(2 * nb_entries)
    X_fm = coo_matrix((data, (rows, cols)), shape=(nb_entries, nb_users + nb_items)).tocsr()

    q_file = X_fm_file.replace('X_fm', 'q')
    if os.path.isfile(q_file):
        q = load_npz(q_file)
        X_fm = hstack((X_fm, q[df['item'] - nb_users])).tocsr()

    if is_regression:
        y_fm = np.array(df['rating'])
    else:
        y_fm = np.array(df['outcome']).astype(np.float32)
    save_npz(X_fm_file, X_fm)
    np.save(DATA_PATH / 'y_fm.npy', y_fm)
else:
    X_fm = load_npz(X_fm_file)


def make_sparse_tf(X_fm):
    nb_rows, _ = X_fm.shape
    rows, cols, data = find(X_fm)
    indices = list(zip(rows, cols))
    return tf.SparseTensorValue(indices, data, [nb_rows, nb_users + nb_items + nb_skills])


nb_skills = X_fm.shape[1] - nb_users - nb_items

# Set dataset indices (folds)
i = {}
try:
    N, M, X_train, X_test, y_train, y_test, i = load_data(DATA)
    # i['trainval'] = pd.read_csv(DATA_PATH / 'trainval.csv')['index'].tolist()
    # i['test'] = pd.read_csv(DATA_PATH / 'test.csv')['index'].tolist()
    nb_users_test = len(i['test'])
    logging.warning('managed to load trainval/test')
    # i['train'], i['valid'] = train_test_split(i['trainval'], test_size=0.2, shuffle=True)
except Exception as e:
    logging.warning('No trainval test %s %s', repr(e), str(e))
    if DATA not in {'fraction', 'movie1M', 'movie10M', 'movie100k'}:
        users = df['userId'].unique()
        items = df['movieId'].unique()
        user_train, user_test = train_test_split(users, test_size=0.2, shuffle=True)
        nb_users_test = len(user_test)
        if options.single_user:
            user_test = user_test[:1]
        if options.split_valid:
            item_train, item_test = train_test_split(items, test_size=0.5, shuffle=True)
        else:
            item_train, item_test = items, items
        i['trainval'] = df.query('userId in @user_train').index
        i['train'] = i['trainval']
        i['valid'] = i['trainval']
        i['test'] = df.query('userId in @user_test').index
        i['ongoing_test'] = []
        # i['test'] = i['trainval']
        i['test_x'] = df.query('userId in @user_test and movieId in @item_train').index
        i['test_y'] = df.query('userId in @user_test and movieId in @item_test').index
        for key in i:
            print(key, max(i[key]) if len(i[key]) else 0, df.shape)

        OVBFM_PATH = Path('../Scalable-Variational-Bayesian-Factorization-Machine')
        to_ovbfm = DATA
        with open(OVBFM_PATH / 'data' / f'{to_ovbfm}.train_libfm', 'w') as f:
            for user, item, rating in np.array(df.loc[i['trainval'], ['user', 'item', 'outcome']]):
                f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, item))

        with open(OVBFM_PATH / 'data' / f'{to_ovbfm}.test_libfm', 'w') as f:
            for user, item, rating in np.array(df.loc[i['test'], ['user', 'item', 'outcome']]):
                f.write('{:d} {:d}:1 {:d}:1\n'.format(rating, user, item))

    else:
        pass
        # i['trainval'], i['test'] = train_test_split(list(range(nb_entries)), test_size=0.2, shuffle=True)
        # i['train'], i['valid'] = train_test_split(i['trainval'], test_size=0.2, shuffle=True)


X = {}
X_sp = {'batch': []}
y = {}
nb_samples = {}
nb_occurrences = {}


def update_vars(category):
    rows = df.iloc[i[category]]
    print(category, rows)
    X[category] = np.array(rows[['user', 'item']])
    print(category, X[category].size)
    if is_regression:
        y[category] = np.array(rows['rating']).astype(np.float32)
    else:
        y[category] = np.array(rows['outcome']).astype(np.float32)
    nb_samples[category] = len(X[category])
    nb_occurrences[category] = X_fm[i[category]].sum(axis=0).A1
    X_sp[category] = make_sparse_tf(X_fm[i[category]])


def softplus(x):
    return np.log(1 + np.exp(x))

np_link = softplus if options.link == 'softplus' else np.abs


initializers = defaultdict(lambda: tf.truncated_normal_initializer(stddev=0.1))
if options.load:
    with open(DATA_PATH / f'saved_folds_weights_{options.d}.pickle', 'rb') as f:
        data = pickle.load(f)
    i = data['folds']  # Override folds
    nb_users_test = data['nb_test_users']
    for var_name in data:
        if var_name != 'folds':
            # Initializing item embeddings
            item_pos = df['item'].min()
            data['other_entities:0'][item_pos:, :embedding_size] = -10.  # Kill those neurons

            # Initializing item biases
            data['other_entities:0'][item_pos:, 1] = -10.
            '''print(data['other_entities:0'][20:, :3].round(3))
                                                print(np_link(data['other_entities:0'][20:, :3].round(3)))
                                                print('checksum', np_link(data['other_entities:0'][20:, :3]).sum())
                                                print('mean', data['other_entities:0'][20:, 3:].round(2))'''
            '''plt.imshow(data['other_entities:0'])
                                                plt.colorbar()
                                                plt.savefig(DATA_PATH / 'item_emb.png')'''

            # Initializing user embeddings
            train_user_ids = df.iloc[i['trainval']]['user'].unique()
            test_user_ids = df.iloc[i['test']]['user'].unique()
            data['user_entities:0'][test_user_ids] = data['user_entities:0'][train_user_ids].mean(axis=0)

            # Initializing user biases
            data['user_biases:0'][test_user_ids] = data['user_biases:0'][train_user_ids].mean(axis=0)

            # sys.exit(0)
            initializers[var_name.replace(':0', '')] = tf.constant_initializer(
                data[var_name])

for category in i:
    update_vars(category)


dt = time.time()

alpha = tf.get_variable(
    'alpha', shape=[], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)

emb_user_mu = tf.get_variable('emb_user_mu', shape=[embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
emb_user_lambda = tf.get_variable(
    'emb_user_lambda', shape=[embedding_size], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)
emb_item_mu = tf.get_variable('emb_item_mu', shape=[embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
emb_item_lambda = tf.get_variable(
    'emb_item_lambda', shape=[embedding_size], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)

bias_user_mu = tf.get_variable('bias_user_mu', shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
bias_user_lambda = tf.get_variable(
    'bias_user_lambda', shape=[], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)
bias_item_mu = tf.get_variable('bias_item_mu', shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
bias_item_lambda = tf.get_variable(
    'bias_item_lambda', shape=[], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)
# OVBFM does not consider mu
'''emb_user_mu = tf.constant([0.] * embedding_size)
emb_item_mu = tf.constant([0.] * embedding_size)
bias_user_mu = tf.constant(0.)
bias_item_mu = tf.constant(0.)'''

global_bias = tf.get_variable('global_bias', shape=[],
    initializer=initializers['global_bias'])
user_entities = tf.get_variable('user_entities',
    shape=[nb_users + nb_items + nb_skills, 2 * embedding_size],
    initializer=initializers['user_entities'])
other_entities = tf.get_variable('other_entities',
    shape=[nb_users + nb_items + nb_skills, 2 * embedding_size],
    initializer=initializers['other_entities'])
user_biases = tf.get_variable('user_biases',
    shape=[nb_users + nb_items + nb_skills, 2],
    initializer=initializers['user_biases'])
other_biases = tf.get_variable('other_biases',
    shape=[nb_users + nb_items + nb_skills, 2],
    initializer=initializers['other_biases'])
all_entities = tf.constant(np.arange(nb_users + nb_items + nb_skills))

def make_mu_bias(lambda_):
    return wtfd.Normal(loc=0., scale=1. / tf.sqrt(lambda_))

def make_mu(lambda_):
    return wtfd.MultivariateNormalDiag(loc=0., scale_diag=1. / tf.sqrt(lambda_))

def make_lambda():
    return wtfd.Gamma(1., 1.)

def make_embedding_prior():
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size, name='emb_prior')

def make_embedding_prior2(mu0, lambda0):
    return wtfd.MultivariateNormalDiag(loc=mu0, scale_diag=1 / tf.sqrt(lambda0), name='hyperprior')

def make_embedding_prior3(priors, entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch, name='priors_prec')
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=1 / tf.sqrt(prior_prec_entity), name='strong_emb_prior')

def make_bias_prior():
    # return tfd.Normal(loc=0., scale=1.)
    return wtfd.Normal(loc=0., scale=1., name='bias_prior')

def make_bias_prior2(mu0, lambda0):
    # return tfd.Normal(loc=0., scale=1.)
    return wtfd.Normal(loc=mu0, scale=1 / tf.sqrt(lambda0))

def make_bias_prior3(priors, entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch)
    return wtfd.Normal(loc=0., scale=1/tf.sqrt(prior_prec_entity[:, 0]), name='strong_bias_prior')

def make_emb_posterior(entities, entity_batch):
    features = tf.nn.embedding_lookup(entities, entity_batch)
    if options.degenerate:
        std_devs = tf.zeros(embedding_size)  # Is it equivalent to std to 0?
    else:
        # std_devs = tf.nn.relu(features[:, :embedding_size])  # Non-differentiable
        std_devs = link(features[:, :embedding_size])  # Differentiable
    return wtfd.MultivariateNormalDiag(loc=features[:, embedding_size:], scale_diag=std_devs, name='emb_posterior')

def make_bias_posterior(biases, entity_batch):
    bias_batch = tf.nn.embedding_lookup(biases, entity_batch)
    if options.degenerate:
        std_dev = 0.
    else:
        std_dev = link(bias_batch[:, 1])
    return wtfd.Normal(loc=bias_batch[:, 0], scale=std_dev, name='bias_posterior')

# def make_item_posterior(item_batch):
#     items = tf.get_variable('items', shape=[nb_items, 2 * embedding_size])
#     feat_items = tf.nn.embedding_lookup(items, item_batch)
#     return tfd.Normal(loc=feat_items[:embedding_size], scale=feat_items[embedding_size:])

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_batch')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_batch')
X_fm_batch = tf.sparse_placeholder(tf.float32, shape=[None, nb_users + nb_items + nb_skills], name='sparse_batch')
outcomes = tf.placeholder(tf.float32, shape=[None], name='outcomes')

emb_user_mu0 = make_mu(emb_user_lambda)  # libFM: \mu_\pi1^v ~ N(0, 1 / \lambda_\pi1^v)
emb_item_mu0 = make_mu(emb_item_lambda)  # libFM: \mu_\pi2^v ~ N(0, 1 / \lambda_\pi2^v)
bias_user_mu0 = make_mu_bias(bias_user_lambda)  # libFM: \mu_\pi1^w ~ N(0, 1 / \lambda_\pi1^w)
bias_item_mu0 = make_mu_bias(bias_item_lambda)  # libFM: \mu_\pi2^w ~ N(0, 1 / \lambda_\pi2^w)

precision_prior = make_lambda()  # libFM: \lambda_* ~ Gamma(\alpha_\lambda, \beta_\lambda) = Gamma(1, 1)
global_bias_prior = make_bias_prior()  # w_0 ~ N(\mu_0, \lambda_0) = N(0, 1)
emb_user_prior = make_embedding_prior2(emb_user_mu, emb_user_lambda)  # v_j ~ N(\mu_\pi1^v, 1 / \lambda_\pi1^v)
emb_item_prior = make_embedding_prior2(emb_item_mu, emb_item_lambda)  # v_j ~ N(\mu_\pi2^v, 1 / \lambda_\pi2^v)
bias_user_prior = make_bias_prior2(bias_user_mu, bias_user_lambda)  # w_j ~ N(\mu_\pi1^v, 1 / \lambda_\pi1^v)
bias_item_prior = make_bias_prior2(bias_item_mu, bias_item_lambda)  # w_j ~ N(\mu_\pi2^v, 1 / \lambda_\pi2^v)

uniq_user_batch, _ = tf.unique(user_batch)
uniq_item_batch, _ = tf.unique(item_batch)

q_uniq_user = make_emb_posterior(user_entities, uniq_user_batch)  # q(v_j) = N(\mu_j^v, 1 / \lambda_j^v)
q_uniq_item = make_emb_posterior(other_entities, uniq_item_batch)
q_uniq_user_bias = make_bias_posterior(user_biases, uniq_user_batch)  # q(w_j) = N(\mu_j^w, 1 / \lambda_j^w)
q_uniq_item_bias = make_bias_posterior(other_biases, uniq_item_batch)

q_user = make_emb_posterior(user_entities, user_batch)  # q(v_j) = N(\mu_j^v, 1 / \lambda_j^v)
q_item = make_emb_posterior(other_entities, item_batch)
q_user_bias = make_bias_posterior(user_biases, user_batch)  # q(w_j) = N(\mu_j^w, 1 / \lambda_j^w)
q_item_bias = make_bias_posterior(other_biases, item_batch)

q_entity = make_emb_posterior(other_entities, all_entities)  # For sparse
q_entity_bias = make_bias_posterior(other_entities, all_entities)
all_bias = q_entity_bias.sample(NB_VARIATIONAL_SAMPLES)
all_feat = q_entity.sample(NB_VARIATIONAL_SAMPLES)

# feat_users2 = tf.nn.embedding_lookup(all_feat, user_batch)
# feat_items2 = tf.nn.embedding_lookup(all_feat, item_batch)
# bias_users2 = tf.nn.embedding_lookup(all_bias, user_batch)
# bias_items2 = tf.nn.embedding_lookup(all_bias, item_batch)

# feat_users = emb_user_prior.sample()
# feat_items = emb_item_prior.sample()
# bias_users = bias_user_prior.sample(tf.shape(user_batch)[0])
# bias_items = bias_item_prior.sample()

if NB_VARIATIONAL_SAMPLES:
    feat_users = q_user.sample(NB_VARIATIONAL_SAMPLES)
    feat_items = q_item.sample(NB_VARIATIONAL_SAMPLES)
    bias_users = q_user_bias.sample(NB_VARIATIONAL_SAMPLES)
    bias_items = q_item_bias.sample(NB_VARIATIONAL_SAMPLES)
else:
    feat_users = q_user.sample()
    feat_items = q_item.sample()
    bias_users = q_user_bias.sample()
    bias_items = q_item_bias.sample()

mean_feat_users = q_user.mean()
mean_feat_items = q_item.mean()
mean_bias_users = q_user_bias.sample()
mean_bias_items = q_item_bias.sample()

# Predictions
def make_likelihood(feat_users, feat_items, bias_users, bias_items):
    logits = global_bias + tf.reduce_sum(feat_users * feat_items, 2) + bias_users + bias_items
    return tfd.Bernoulli(logits)

def make_likelihood_reg(feat_users, feat_items, bias_users, bias_items):
    logits = global_bias + tf.reduce_sum(feat_users * feat_items, 2) + bias_users + bias_items
    # logits = global_bias + tf.diag(feat_users @ tf.transpose(feat_items)) + bias_users + bias_items
    return tfd.Normal(logits, scale=1 / tf.sqrt(alpha), name='pred')

def make_sparse_pred(x):
    #x = tf.cast(x, tf.float32)
    x2 = x# ** 2  # FIXME if x is 0/1 it's okay
    this_bias = tf.reduce_sum(all_bias, axis=0)
    this_feat = tf.reduce_sum(all_feat, axis=0)
    w = tf.reshape(this_bias, (-1, 1))
    V = this_feat
    V2 = V ** 2
    logits = (tf.squeeze(tf.sparse_tensor_dense_matmul(x, w)) +
              0.5 * tf.reduce_sum(tf.sparse_tensor_dense_matmul(x, V) ** 2 -
                                  tf.sparse_tensor_dense_matmul(x2, V2), axis=1))
    return tfd.Bernoulli(logits)

def make_sparse_pred_reg(x):
    #x = tf.cast(x, tf.float32)
    x2 = x# ** 2  # FIXME if x is 0/1 it's okay
    this_bias = tf.reduce_sum(all_bias, axis=0)
    this_feat = tf.reduce_sum(all_feat, axis=0)
    w = tf.reshape(this_bias, (-1, 1))
    # w = tf.reshape(bias[:, 0], (-1, 1))  # Otherwise tf.matmul is crying
    # V = users[:, embedding_size:]
    V = this_feat
    V2 = V ** 2
    logits = (tf.squeeze(tf.sparse_tensor_dense_matmul(x, w)) +
              0.5 * tf.reduce_sum(tf.sparse_tensor_dense_matmul(x, V) ** 2 -
                                  tf.sparse_tensor_dense_matmul(x2, V2), axis=1))
    return tfd.Normal(logits, scale=1 / alpha)

def define_variables(train_category, priors, batch_size, var_list=None):
    global emb_user_prior, emb_item_prior, bias_user_prior, bias_item_prior
    if options.degenerate:
        emb_user_prior = make_embedding_prior()
        emb_item_prior = make_embedding_prior()
        emb_entity_prior = make_embedding_prior()
        bias_user_prior = make_bias_prior()
        bias_item_prior = make_bias_prior()
        bias_entity_prior = make_bias_prior()
    else:
        # emb_user_prior = make_embedding_prior3(priors, user_batch)
        # emb_item_prior = make_embedding_prior3(priors, item_batch)
        emb_entity_prior = make_embedding_prior3(priors, all_entities)
        # bias_user_prior = make_bias_prior3(priors, user_batch)
        # bias_item_prior = make_bias_prior3(priors, item_batch)
        bias_entity_prior = make_bias_prior3(priors, all_entities)

    user_rescale = tf.nn.embedding_lookup(priors, user_batch)[:, 0]
    item_rescale = tf.nn.embedding_lookup(priors, item_batch)[:, 0]
    uniq_user_rescale = tf.clip_by_value(tf.nn.embedding_lookup(priors, uniq_user_batch)[:, 0], 1, 100000000)
    uniq_item_rescale = tf.clip_by_value(tf.nn.embedding_lookup(priors, uniq_item_batch)[:, 0], 1, 100000000)
    entity_rescale = priors[:, 0]

    if is_classification:
        likelihood = make_likelihood(feat_users, feat_items, bias_users, bias_items)
        sparse_pred = make_sparse_pred(X_fm_batch)
    else:
        likelihood = make_likelihood_reg(feat_users, feat_items, bias_users, bias_items)
        sparse_pred = make_sparse_pred_reg(X_fm_batch)
    pred2 = sparse_pred.mean()
    # ll = make_likelihood(feat_users2, feat_items2, bias_users2, bias_items2)
    pred = tf.reduce_mean(likelihood.mean(), axis=0)  # À cause du nombre de variational samples
    # pred = tf.reduce_mean(likelihood.sample(), axis=0)  # Sampling instead of mean is worse
    pred_mean = global_bias + tf.reduce_sum(mean_feat_users * mean_feat_items, 1) + mean_bias_users + mean_bias_items
    if is_classification:
        pred_mean = tf.sigmoid(pred_mean)
    else:
        pred = tf.clip_by_value(pred, 1, 5)
    # pred_mean = pred in regression but not in classification

    # likelihood_var = make_likelihood_reg(sigma2, q_user, q_item, q_user_bias, q_item_bias)
    # print(likelihood.log_prob([1, 0]))

    # Check shapes
    # print('likelihood', likelihood.log_prob(outcomes))
    # print('prior', emb_user_prior.log_prob(feat_users))
    # print('scaled prior', emb_user_prior.log_prob(feat_users) / user_rescale)
    # print('posterior', q_user.log_prob(feat_users))
    # print('bias prior', bias_user_prior.log_prob(bias_users))
    # print('bias posterior', q_user_bias.log_prob(bias_users))

    # sentinel = likelihood.log_prob(outcomes)
    # sentinel = bias_prior.log_prob(bias_users)
    # sentinel = tf.reduce_sum(ll.log_prob(outcomes))
    # sentinel2 = tf.reduce_sum(likelihood.log_prob(outcomes))

    # elbo = tf.reduce_mean(
    #     user_rescale * item_rescale * likelihood.log_prob(outcomes) +
    #     item_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
    #                     emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
    #     user_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
    #                     emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)))

    # (nb_users + nb_items) / 2
    if options.degenerate:
        # elbo = -(tf.reduce_sum((pred - outcomes) ** 2 / 2) +
        #          0.1 * tf.reduce_sum(tf.nn.l2_loss(bias_users) + tf.nn.l2_loss(bias_items) +
        #          tf.nn.l2_loss(feat_users) + tf.nn.l2_loss(feat_items)))
        elbo = tf.reduce_mean()
        '''likelihood.log_prob(outcomes) - # - car KL
                # bias_user_prior.log_prob(bias_users) + emb_user_prior.log_prob(feat_users)
                1 / user_rescale * (wtfd.kl_divergence(q_user, emb_user_prior) +
                                    emb_user_mu0.log_prob(emb_user_mu) + precision_prior.log_prob(emb_user_lambda) +
                                    bias_user_mu0.log_prob(bias_user_mu) + precision_prior.log_prob(bias_user_lambda)) +
                1 / item_rescale * (bias_item_prior.log_prob(bias_items) + emb_item_prior.log_prob(feat_items) +
                                    emb_item_mu0.log_prob(emb_item_mu) + precision_prior.log_prob(emb_item_lambda) +
                                    bias_item_mu0.log_prob(bias_item_mu) + precision_prior.log_prob(bias_item_lambda)), name='elbo')'''
    # / 2 : 1.27
    # * 2 : 1.16
    elif options.sparse:
        nb_occ = tf.sparse_reshape(tf.sparse_reduce_sum_sparse(X_fm_batch, axis=0), (1, -1))

        lp_lq = tf.reduce_sum(bias_entity_prior.log_prob(all_bias) - q_entity_bias.log_prob(all_bias) +
                              emb_entity_prior.log_prob(all_feat) - q_entity.log_prob(all_feat), axis=0)
        nonzero_entity_rescale = 1 + tf.maximum(0., entity_rescale - 1)
        lp_lq = tf.reshape(lp_lq / nonzero_entity_rescale, (-1, 1))
        relevant_scaled_lp_lq = tf.squeeze(tf.sparse_tensor_dense_matmul(nb_occ, lp_lq))

        elbo = (tf.reduce_mean(sparse_pred.log_prob(outcomes)) +
                relevant_scaled_lp_lq / batch_size)

    else:
        # elbo = tf.reduce_mean(
        #     nb_samples[train_category] * likelihood.log_prob(outcomes) +
        #     nb_samples[train_category] * 1 / user_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
        #                       emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
        #     nb_samples[train_category] * 1 / item_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
        #                       emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')

        ''' - tf.reduce_sum(wtfd.kl_divergence(q_uniq_user_bias, bias_user_prior) +
                            wtfd.kl_divergence(q_uniq_user, emb_user_prior)) +
            - tf.reduce_sum(wtfd.kl_divergence(q_uniq_item_bias, bias_item_prior) +
                            wtfd.kl_divergence(q_uniq_item, emb_item_prior))'''

        # Objective function ELBO
        logging.warning('parameters %d %d %d', nb_samples[train_category], nb_users, nb_iters)
        elbo = (nb_samples[train_category] * (
            tf.reduce_mean(
                likelihood.log_prob(outcomes)
            ))
        # nb_samples[train_category] / (nb_users) * nb_iters
        - nb_samples[train_category] * tf.reduce_mean(
            # uniq_user_rescale
            1 / uniq_user_rescale * (wtfd.kl_divergence(q_uniq_user_bias, bias_user_prior) +
                                wtfd.kl_divergence(q_uniq_user, emb_user_prior))
        )
        # nb_samples[train_category] / (nb_items) * nb_iters
        - nb_samples[train_category] * tf.reduce_mean(
            # uniq_item_rescale
            1 / uniq_item_rescale * (wtfd.kl_divergence(q_uniq_item_bias, bias_item_prior) +
                                wtfd.kl_divergence(q_uniq_item, emb_item_prior))
        )
        - (
            global_bias_prior.log_prob(global_bias)
            + emb_user_mu0.log_prob(emb_user_mu) + tf.reduce_sum(precision_prior.log_prob(emb_user_lambda))
            + emb_item_mu0.log_prob(emb_item_mu) + tf.reduce_sum(precision_prior.log_prob(emb_item_lambda))
            + bias_user_mu0.log_prob(bias_user_mu) + precision_prior.log_prob(bias_user_lambda)
            + bias_item_mu0.log_prob(bias_item_mu) + precision_prior.log_prob(bias_item_lambda)
            + precision_prior.log_prob(alpha)
        )) #/ nb_samples[train_category]

        # many_feats = q_user.sample(100)

    sentinel = {
        # 'nb outcomes': tf.shape(outcomes),
        # 'nb samples': tf.constant(nb_samples[train_category]),
        # 'user batch': user_batch,
        # 'uniq_user': uniq_user_rescale,
        # 'uniq_item': uniq_item_rescale,
        'user_batch': tf.shape(user_batch),
        'user unique batch': tf.shape(tf.unique(user_batch)[0]),
        # 'nb_occ': nb_occ,
        # 'p - q': emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users),
        # 'many p - q': tf.reduce_mean(emb_user_prior.log_prob(many_feats) - q_user.log_prob(many_feats), axis=0),
        # 'kl attendu': -wtfd.kl_divergence(q_user, emb_user_prior),
        # 'kl attendu aussi': -wtfd.kl_divergence(q_user_bias, bias_user_prior),
        # 'lplq': relevant_scaled_lp_lq,
        # 'batch ll log prob': nb_samples['train'] * tf.reduce_mean(likelihood.log_prob(outcomes)),
        # 'll log prob shape': tf.shape(likelihood.log_prob(outcomes)),
        # 'user_rescale shape': tf.shape(user_rescale),
        # 'batch p - q emb user': tf.reduce_sum(1 / user_rescale * tmp),
        # 'p - q emb user shape': tf.shape(tmp),
        # 'prec bias user lambda': precision_prior.log_prob(bias_user_lambda),
        # 'prec bias user lambda shape': tf.shape(precision_prior.log_prob(bias_user_lambda)),
        # 'prec emb user lambda': precision_prior.log_prob(emb_user_lambda),
        # 'prec emb user lambda shape': tf.shape(precision_prior.log_prob(emb_user_lambda)),
        # 'll log prob sparse': -sparse_pred.log_prob(outcomes),
        # 'll log prob has nan': tf.reduce_any(tf.is_nan(likelihood.log_prob(outcomes))),
        # 'll log prob sparse has nan': tf.reduce_any(tf.is_nan(sparse_pred.log_prob(outcomes))),
        # 's ll log prob': -tf.reduce_sum(likelihood.log_prob(outcomes)),
        # 's pred delta': tf.reduce_sum((pred - outcomes) ** 2 / 2 + np.log(2 * np.pi) / 2),
        # 'entity_rescale sum': tf.reduce_sum(entity_rescale),
        # 'nb occ sum': tf.constant(nb_occurrences[train_category].sum()),
        # 'logits': logits,
        # 'max logits': tf.reduce_max(logits),
        # 'min logits': tf.reduce_min(logits),
        # 'max logits2': tf.reduce_max(logits2),
        # 'min logits2': tf.reduce_min(logits2),
        # 'bias sample': bias_users[0],
        # 'bias log prob': -bias_user_prior.log_prob(bias_users)[0],
        # 'sum bias log prob': -tf.reduce_sum(bias_user_prior.log_prob(bias_users)),
        # 'likelihood (outcomes)': tf.shape(likelihood.log_prob(outcomes)),
        # 'pred': pred,
        # 'pred shape': tf.shape(pred),
        # 'mean pred': pred_mean,
        # 'mean pred shape': tf.shape(pred_mean),
        # 'pred2': pred2,
        # 'max pred': tf.reduce_max(pred),
        # 'min pred': tf.reduce_min(pred),
        # 'max pred2': tf.reduce_max(pred2),
        # 'min pred2': tf.reduce_min(pred2),
        # 'has nan': tf.reduce_any(tf.is_nan(pred2))
        # 'bias mean': bias_user_prior.mean(),
        # 'bias delta': bias_users[0] ** 2 / 2 + np.log(2 * np.pi) / 2,
        # 'sum bias delta': tf.reduce_sum(bias_users ** 2 / 2 + np.log(2 * np.pi) / 2)
    }

    if var_list is not None:
        infer_op = optimizer.minimize(-elbo, var_list=var_list)
    else:
        infer_op = optimizer.minimize(-elbo)
    if options.sparse:
        return infer_op, elbo, pred2, likelihood, sentinel
    else:
        return infer_op, elbo, pred, likelihood, sentinel

# elbo4 = (# nb_samples['train'] * tf.reduce_mean(ll.log_prob(outcomes)) +
#         nb_samples['train'] * sparse_pred.log_prob(outcomes) +
#         tf.reduce_sum(bias_user_prior.log_prob(all_bias) - q_entity_bias.log_prob(all_bias)) +
#         tf.reduce_sum(emb_user_prior.log_prob(all_feat) - q_entity.log_prob(all_feat)))
# elbo4 = tf.add(elbo4, 0, name='elbo4')

# elbo2 = tf.reduce_mean(
#     nb_samples['train'] * likelihood.log_prob(outcomes) +
#                      (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
#                       emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
#                      (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
#                       emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo2')

optimizer = tf.train.AdamOptimizer(learning_rate)  # 0.001
# optimizer = tf.train.GradientDescentOptimizer(gamma)  # 0.001
# optimizer = tf.train.MomentumOptimizer(gamma, momentum=0.9)  # 0.001
# optimizer = tf.train.RMSPropOptimizer(gamma)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def avgstd(l):
    '''
    Given a list of values, returns a 95% confidence interval
    if the standard deviation is unknown.
    '''
    n = len(l)
    mean = sum(l) / n
    if n == 1:
        return '%.3f' % round(mean, 3)
    std_err = sem(l)
    confidence = 0.95
    h = std_err * sqrt(n) * t.ppf((1 + confidence) / 2, n - 1)
    return 'σ(%.3f ± %.3f) = [%.3f, %.3f]' % (mean, h, sigmoid(mean - h), sigmoid(mean + h))


def make_feed(category):
    return {user_batch: X[category][:, 0],
            item_batch: X[category][:, 1],
            outcomes: y[category]}#,
            # X_fm_batch: X_sp[category]}


class VFM:
    def __init__(self, train_category, valid_category=None, test_category='test',
                 stop_when_worse=('train', 'elbo'), nb_batches=1,
                 train_patience=options.train_patience,
                 valid_patience=options.valid_patience, optimized_vars=None):
        self.start_time = time.time()
        if valid_category is None:
            valid_category = train_category
        self.data = {
            'train': train_category,
            'valid': valid_category,
            'test': test_category
        }
        self.metrics = {
            'train': defaultdict(list),
            'valid': defaultdict(list),
            'test': defaultdict(list),
            '': defaultdict(list),
            'random': defaultdict(list),
            'mean': defaultdict(list),
            'variance': defaultdict(list),
            'time': {}
        }
        self.category_watcher, self.metric_watcher = stop_when_worse
        self.train_patience = train_patience
        self.valid_patience = valid_patience
        self.optimized_vars = optimized_vars
        self.nb_batches = nb_batches
        self.strategy = ''
        self.all_preds = defaultdict(list)
        self.init_train()

    def reset(self, strategy='random'):
        self.strategy = strategy
        # self.metrics['train'] = defaultdict(list)
        # self.metrics['test'] = defaultdict(list)
        i[self.data['train']] = []  # Reset training set
        update_vars(self.data['train'])

    def model_name(self):
        if options.degenerate:
            title = 'fm-map'
        elif options.sparse:
            title = 'vfm-sparse'
        else:
            title = 'vfm'
        name = '{:s}-{:s}-{:s}-{:s}-{:d}'.format(
            DATA, title, self.data['train'], self.strategy, options.d)
        if SUFFIX:
            name += '-' + SUFFIX
        return name

    def __getstate__(self):
        state = self.__dict__.copy()
        '''for key, value in state.items():
                                    logging.warning('hiya %s %s', key, type(value))'''
        del state['start_time']
        del state['pred'], state['elbo'], state['likelihood'], state['infer_op']
        return state

    def save_model(self):
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        data = {'folds': i, 'nb_test_users': nb_users_test}
        for var, val in zip(tvars, tvars_vals):
            tf.add_to_collection('vars', var)
            data[var.name] = val
            # logging.warning('%s %s %s', var.name, val.shape, val)  # Prints the name of the variable alongside its value.

        with open(DATA_PATH / f'saved_folds_weights_{options.d}.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_PATH / 'vfm.pickle', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open(DATA_PATH / 'vfm.pickle', 'rb') as f:
            self = pickle.load(f)

    def stopping_rule(self):
        K = 5
        if self.category_watcher == 'train':
            last_values = self.metrics[self.category_watcher][self.metric_watcher][-self.train_patience:]
            is_decreasing = self.metric_watcher in {'acc', 'auc', 'elbo'}  # If these metrics decrease, it's worse
            return (
                self.epoch >= MAX_EPOCHS or (
                    self.epoch >= MIN_EPOCHS and
                    len(last_values) >= self.train_patience and
                    last_values == sorted(last_values, reverse=is_decreasing)
                ), last_values)
        elif self.epoch < K - 1 or self.epoch % self.valid_patience != 0:
            return False, []
        else:  # Try tracking quotient
            train = self.metrics['train']['elbo']
            strip = train[-K:]
            progress = 1000 * (sum(strip) / (K * max(strip)) - 1)
            valid = self.metrics['valid'][self.metric_watcher]  # Most possibly RMSE
            print('valid', valid)
            gen_loss = 100 * (valid[-1] / min(valid) - 1)
            quotient = gen_loss / progress
            return quotient > 0.2, valid[-2:]

    def save_metrics(self, category, epoch, y_truth, y_pred):
        if VERBOSE:
            print('[%s] pred' % category, y_truth[:5], y_pred[:5])
        if len(self.metrics[category]['epoch']) == 0 or self.metrics[category]['epoch'][-1] != epoch:
            self.metrics[category]['epoch'].append(epoch)
        self.all_preds[category].append(y_pred.tolist())
        # print(np.array(self.all_preds).shape, [len(term) for term in self.all_preds])
        mean_pred = np.array(self.all_preds[category]).mean(axis=0)
        self.metrics[category]['acc'].append(np.mean(y_truth == np.round(y_pred)))
        self.metrics[category]['acc_all'].append(np.mean(y_truth == np.round(mean_pred)))
        # print('shape', mean_pred.shape)
        # sys.exit(0)
        if set(y_truth) == {0., 1.}:
            # from collections import Counter
            # print(Counter(y_truth))
            self.metrics[category]['auc'].append(roc_auc_score(y_truth, y_pred))
            self.metrics[category]['map'].append(average_precision_score(y_truth, y_pred))
            self.metrics[category]['nll'].append(log_loss(y_truth, y_pred, eps=1e-6))
        else:
            self.metrics[category]['rmse'].append(mean_squared_error(y_truth, y_pred) ** 0.5)
            self.metrics[category]['rmse_all'].append(mean_squared_error(y_truth, mean_pred) ** 0.5)
        if VERBOSE:
            print('[%s] ' % category + ' '.join('{:s}={}'.format(metric, np.round(self.metrics[category][metric][-1], 3)) for metric in self.metrics[category]))

    def run_and_save(self, category):
        # print('makefeed', category, self.data[category], make_feed(self.data[category]))
        valid_pred = sess.run(self.pred, feed_dict=make_feed(self.data[category]))
        # print(sess.run(entity))
        # print(sess.run(bias))
        # print(sess.run(global_bias))
        # print(valid_pred)
        self.save_metrics(category, self.epoch, y[self.data[category]], valid_pred)

    def save_logs(self):
        filename = '{:s}-{:d}.txt'.format(self.model_name(), int(round(time.time())))
        self.metrics['model_name'] = self.model_name()
        # self.metrics['train']['elbo'] = self.metrics['train']['elbo'].astype()
        self.metrics['time']['total'] = time.time() - self.start_time
        data = {
            'description': DESCRIPTION,
            'date': datetime.now().isoformat(),
            'stopped': '{:d}/{:d}'.format(self.epoch, MAX_EPOCHS),
            'args': vars(options),
            'metrics': self.metrics,
        }
        save_to_path = Path('results') / filename
        plot_after(data, save_to_path)
        with open(save_to_path, 'w') as f:
            f.write(json.dumps(data, indent=4))
        logging.warning('execute python rule.py %s', save_to_path)
        os.system(f'python rule.py {save_to_path}')

    def plot(self, category):
        plt.clf()
        for metric in self.metrics[category]:
            if metric != 'epoch':
                x = np.arange(len(self.metrics[category][metric]))
                plt.plot(x, self.metrics[category][metric], label=metric)
        plt.legend()
        plt.savefig(DATA_PATH / f'{self.model_name()}_{category}.png')

    def select_next_question(self, pool_name, nb_questions=1, strategy='random'):
        # For each user
        unasked = set(i[pool_name]) - set(i[self.data['train']])
        df_unasked = df.iloc[list(unasked)].reset_index()
        remaining_dict = {user_batch: df_unasked['user'],
                          item_batch: df_unasked['item']}
        df_unasked['proba_means'], df_unasked['logit_variances'] = (
            self.predict_proba(remaining_dict))
        df_unasked['certainty'] = abs(df_unasked['proba_means'] - 0.5)    
        df_unasked['random'] = np.random.random(size=(len(unasked),))

        if strategy == 'random':
            selection = df_unasked.sort_values('random').groupby('user').first()
        elif strategy == 'mean':
            selection = df_unasked.sort_values('certainty').groupby('user').first()
        elif strategy == 'variance':
            selection = df_unasked.sort_values(
                'logit_variances', ascending=False).groupby('user').first()
        i[self.data['train']].extend(selection['index'])    
        '''logging.warning('dataset is now %s %s', i[self.data['train']],
                                                X[self.data['train']])'''
        update_vars(self.data['train'])

    def predict_proba(self, feed_dict=make_feed('test')):
        likelihood_samples, logits = sess.run([
            self.likelihood.mean(), self.likelihood.logits],
            feed_dict=feed_dict)
        proba_means = likelihood_samples.mean(axis=0)
        logit_variances = logits.var(axis=0)
        '''plt.clf()
                                plt.hist(logits[:, 0], bins=50)
                                plt.savefig('q1.png')'''
        # print(feed_dict)
        return proba_means, logit_variances

    def init_train(self, training_set_name=None):
        if training_set_name is None:
            training_set_name = self.data['train']

        self.nb_batches = max(1, min(nb_samples[self.data['train']], self.nb_batches))
        self.batch_size = nb_samples[self.data['train']] // self.nb_batches

        priors = tf.clip_by_value(
            tf.constant(nb_occurrences[training_set_name][:, None].repeat(
                embedding_size, axis=1), dtype=np.float32), 1, 1000000)
        with tf.variable_scope(self.model_name()):
            self.infer_op, self.elbo, self.pred, self.likelihood, self.sentinel = define_variables(
                training_set_name, priors, self.batch_size,
                self.optimized_vars)

        if self.optimized_vars is None or options.load:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.variables_initializer(self.optimized_vars))

    def train(self, training_set_name=None):
        if training_set_name is None:
            training_set_name = self.data['train']
        self.init_train(training_set_name)

        self.epoch = 0
        self.metrics['train'] = defaultdict(list)  # Flush metrics
        self.metrics['test'] = defaultdict(list)

        logging.warning('Test initial performance')
        self.run_and_save('test')

        if is_classification:
            self.predict_proba()

        while True:
            self.epoch += 1

            dt = time.time()
            train_elbos = []
            all_ids = np.arange(len(i[training_set_name]))
            np.random.shuffle(all_ids)
            for nb_iter in range(self.nb_batches):
                batch_ids = all_ids[nb_iter * self.batch_size:(nb_iter + 1) * self.batch_size]
                # print(nb_iter, self.batch_size, len(all_ids), len(batch_ids), batch_ids)
                X['batch'] = X[training_set_name][batch_ids]
                y['batch'] = y[training_set_name][batch_ids]
                # X_sp['batch'] = make_sparse_tf(X_fm[i[training_set_name]][batch_ids])

                # print('trois elbo', sess.run([elbo, elbo, elbo], feed_dict=make_feed('batch')))
                if options.method == 'adam':
                    _, train_elbo = sess.run([self.infer_op, self.elbo], feed_dict=make_feed('batch'))
                else:
                    scipy_optimizer = tf.contrib.opt.ScipyOptimizerInterface(-elbo, method='L-BFGS-B', var_list=self.optimized_vars)
                    scipy_optimizer.minimize(sess, feed_dict=make_feed('batch'))
                    train_elbo = sess.run(elbo, feed_dict=make_feed('batch'))

                if VERBOSE >= 100:
                    print(self.sentinel)
                    for key in self.sentinel:
                        print(key)
                        val = sess.run(self.sentinel[key], feed_dict=make_feed('batch'))
                        print(key, val)
                    # sys.exit(0)

                assert np.isnan(train_elbo) == False
                train_elbos.append(train_elbo)
                if nb_iter == 0:
                    self.metrics['time']['per_batch'] = time.time() - dt
            if self.epoch == 1:
                self.metrics['time']['per_epoch'] = time.time() - dt

            if self.data['valid'] == 'valid' and self.epoch % self.valid_patience == 0:
                self.run_and_save('valid')

            self.metrics['train']['epoch'].append(self.epoch)
            self.metrics['train']['elbo'].append(np.mean(train_elbos).astype(np.float64))

            has_to_stop, watched_values = self.stopping_rule()

            if VERBOSE >= 10:
                self.run_and_save('train')

            if self.epoch % COMPUTE_TEST_EVERY == 0 or has_to_stop:
                self.run_and_save('test')

            if VERBOSE:
                print('{:.3f}s [{}] Epoch {}: Lower bound = {}'.format(time.time() - dt, self.model_name(), self.epoch, self.metrics['train']['elbo'][-1]))

            if has_to_stop:
                break

        print('Stop training: {:s} {:s} is {:s}'.format(self.category_watcher, self.metric_watcher, str(watched_values)))

        logit_variances = None
        if is_classification:
            _, logit_variances = self.predict_proba()

        logging.warning('Test contains %d samples', nb_samples[self.data['test']])
        self.metrics[self.strategy]['nb_train_samples'].append(int(nb_samples[self.data['train']]))  #  // nb_users_test
        if logit_variances is not None:
            self.metrics[self.strategy]['mean test variance'].append(logit_variances.mean().astype(np.float64))
        for metric in self.metrics['test']:
            final = self.metrics['test'][metric][-1]
            best = (max if metric in {'auc', 'acc', 'epoch', 'map'} else min)(
                self.metrics['test'][metric])
            self.metrics['final ' + metric] = final
            self.metrics[self.strategy][metric].append(final)
            self.metrics['best ' + metric] = float(best)
            self.metrics[self.strategy]['best ' + metric].append(best)
            print('[{:s}] final={:f} best={:f}'.format(metric, final, best))
        # sys.exit(0)

        self.save_logs()
        self.plot('train')
        self.plot('test')

        # return self.metrics[self.category_watcher][self.metric_watcher][-self.train_patience]  # Best metric reported so far
        return min(self.metrics[self.category_watcher][self.metric_watcher][-2:])  # Min valid RMSE of last 2 steps


if __name__ == '__main__':
    stopped_at = None
    time_per_batch = None
    time_per_epoch = None
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_writer = tf.summary.FileWriter('/tmp/test', sess.graph)

        best_sigma = 0.
        '''if is_classification:
                                    best_sigma = 0.
                                elif options.sigma2 != -1:
                                    best_sigma = options.sigma2
                                else:  # Have to find sigma2 via cross validation
                                    sigma2s = [0.1, 0.2, 0.5, 1.]
                                    valid_metrics = []
                                    for sigma2 in sigma2s:
                                        vfm = VFM('train', 'valid', sigma2, stop_when_worse=('valid', 'auc' if is_classification else 'rmse'))
                                        valid_metric = vfm.train()
                                        valid_metrics.append(valid_metric)
                                    print('Candidates', dict(zip(sigma2s, valid_metrics)))
                                    best_sigma = sigma2s[np.argmin(valid_metrics)]'''

        # saver = tf.train.Saver()

        if not options.load:
            refit = VFM('trainval', nb_batches=options.nb_batches)
            refit.train()
            refit.save_model()
        else:
            logging.warning('checksum %f', data['other_entities:0'].sum())

        if options.interactive:
            interactive = VFM('ongoing_test',
                optimized_vars=[user_entities, user_biases])

            for strategy in ['random', 'mean', 'variance']:
                interactive.reset(strategy=strategy)

                for q in range(N_QUESTIONS_ASKED):
                    interactive.select_next_question('test_x')
                    if (q + 1) % TRAIN_EVERY_N_QUESTIONS == 0:
                        interactive.train()

        # embeddings = sess.run(other_entities)
        # logging.warning('checksum %f', embeddings.sum())

    print('Finish', time.time() - start_time)
