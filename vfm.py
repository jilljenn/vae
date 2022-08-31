from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from scipy.sparse import coo_matrix, load_npz, save_npz, hstack, find
from tensorflow.python import debug as tf_debug
from collections import Counter, defaultdict
from datetime import datetime
import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow_probability as tfp
wtfd = tfp.distributions
import argparse
import os.path
import getpass
import logging
import pandas as pd
import numpy as np
import yaml
import json
import time
import sys


DESCRIPTION = 'Rescaled mode and test every step'
SUFFIX = 'forced'
start_time = time.time()

parser = argparse.ArgumentParser(description='Run VFM')
parser.add_argument('data', type=str, nargs='?', default='fraction')
parser.add_argument('--degenerate', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--sparse', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--regression', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--classification', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--valid_patience', type=int, nargs='?', default=10)
parser.add_argument('--train_patience', type=int, nargs='?', default=4)
parser.add_argument('--d', type=int, nargs='?', default=3)
parser.add_argument('--gamma', type=float, nargs='?', default=0.01)
parser.add_argument('--sigma2', type=float, nargs='?', default=-1)
parser.add_argument('--nb_batches', type=int, nargs='?', default=1)

parser.add_argument('--v', type=int, nargs='?', default=1) # Verbose
options = parser.parse_args()
 
if getpass.getuser() == 'jj':
    PATH = '/home/jj/code'
else:
    PATH = '/Users/jilljenn/code'

DATA = options.data
print('Data is', DATA)
VERBOSE = options.v
NB_SAMPLES = 1
COMPUTE_TEST_EVERY = 1

# Load data
if DATA in {'fraction', 'movie1M', 'movie10M', 'movie100k'}:
    df = pd.read_csv(os.path.join(PATH, 'vae/data', DATA, 'data.csv'))
    print('Starts at', df['user'].min(), df['item'].min())
    try:
        with open(os.path.join(PATH, 'vae/data', DATA, 'config.yml')) as f:
            config = yaml.load(f)
            nb_users = config['nb_users']
            nb_items = config['nb_items']
    except IOError:
        nb_users = 1 + df['user'].max()
        nb_items = 1 + df['item'].max()
    df['item'] += nb_users
    print(df.head())
else:
    print('Please use an available dataset')
    assert False

# Is it classification or regression?
if options.regression or 'rating' in df:
    is_regression = True
    is_classification = False
elif options.classification or 'outcome' in df:
    is_classification = True
    is_regression = False

nb_entries = len(df)

# Build sparse features
X_fm_file = os.path.join(PATH, 'vae/data', DATA, 'X_fm.npz')
if not os.path.isfile(X_fm_file):
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
    np.save(os.path.join(PATH, 'vae/data', DATA, 'y_fm.npy'), y_fm)
else:
    X_fm = load_npz(X_fm_file)

nb_skills = X_fm.shape[1] - nb_users - nb_items

i = {}
try:
    i['trainval'] = pd.read_csv(os.path.join(PATH, 'vae/data', DATA, 'trainval.csv'))['index'].tolist()
    i['test'] = pd.read_csv(os.path.join(PATH, 'vae/data', DATA, 'test.csv'))['index'].tolist()
except Exception as e:
    logging.warning('No trainval test %s %s', repr(e), str(e))
    i['trainval'], i['test'] = train_test_split(list(range(nb_entries)), test_size=0.2, shuffle=True)
i['train'], i['valid'] = train_test_split(i['trainval'], test_size=0.2, shuffle=True)
data = {key: df.iloc[i[key]] for key in {'train', 'valid', 'trainval', 'test'}}

X = {}
# X_sp = {}
X_sp = {'batch': []}
y = {}
nb_samples = {}
nb_occurrences = {
    'train': X_fm[i['train']].sum(axis=0).A1,
    'trainval': X_fm[i['trainval']].sum(axis=0).A1
}

def make_sparse_tf(X_fm):
    nb_rows, _ = X_fm.shape
    rows, cols, data = find(X_fm)
    indices = list(zip(rows, cols))
    return tf.SparseTensorValue(indices, data, [nb_rows, nb_users + nb_items + nb_skills])

for category in data:
    X[category] = np.array(data[category][['user', 'item']])
    print(category, X[category].size)
    if is_regression:
        y[category] = np.array(data[category]['rating']).astype(np.float32)
    else:
        y[category] = np.array(data[category]['outcome']).astype(np.float32)
    nb_samples[category] = len(X[category])
    X_sp[category] = make_sparse_tf(X_fm[i[category]])

# Config
MAX_EPOCHS = 500
print('Nb samples', nb_samples['train'])
embedding_size = options.d
nb_iters = options.nb_batches
print('Nb iters', nb_iters)
gamma = options.gamma  # gamma 0.001 works better for classification

dt = time.time()

alpha = tf.get_variable(
    'alpha', shape=[], initializer=tf.random_uniform_initializer(minval=0., maxval=1.), constraint=tf.math.abs)
# alpha = tf.constant(options.sigma2)

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

global_bias = tf.get_variable('global_bias', shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
entity = tf.get_variable('entities', shape=[nb_users + nb_items + nb_skills, 2 * embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
bias = tf.get_variable('bias', shape=[nb_users + nb_items + nb_skills, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
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

def make_emb_posterior(user_batch):
    feat_users = tf.nn.embedding_lookup(entity, user_batch)
    # return tfd.Normal(loc=feat_users[:, :embedding_size], scale=feat_users[:, embedding_size:])
    if options.degenerate:
        std_devs = tf.zeros(embedding_size)
    else:
        # 1/tf.sqrt(prior_prec_entity)  # More precise if more ratings
        # tf.ones(embedding_size)  # Too imprecise
        std_devs = tf.nn.softplus(feat_users[:, :embedding_size])
        # std_devs = tf.nn.relu(feat_users[:, :embedding_size])  # Non differentiable
    return wtfd.MultivariateNormalDiag(loc=feat_users[:, embedding_size:], scale_diag=std_devs, name='emb_posterior')

def make_bias_posterior(entity_batch):
    bias_batch = tf.nn.embedding_lookup(bias, entity_batch)
    if options.degenerate:
        std_dev = 0.
    else:
        # 1/tf.sqrt(prior_prec_entity[:, 0])  # More precise if more ratings, should be clipped
        # 1.  # Too imprecise
        std_dev = tf.nn.softplus(bias_batch[:, 1])
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

q_uniq_user = make_emb_posterior(uniq_user_batch)  # q(v_j) = N(\mu_j^v, 1 / \lambda_j^v)
q_uniq_item = make_emb_posterior(uniq_item_batch)
q_uniq_user_bias = make_bias_posterior(uniq_user_batch)  # q(w_j) = N(\mu_j^w, 1 / \lambda_j^w)
q_uniq_item_bias = make_bias_posterior(uniq_item_batch)

q_user = make_emb_posterior(user_batch)  # q(v_j) = N(\mu_j^v, 1 / \lambda_j^v)
q_item = make_emb_posterior(item_batch)
q_user_bias = make_bias_posterior(user_batch)  # q(w_j) = N(\mu_j^w, 1 / \lambda_j^w)
q_item_bias = make_bias_posterior(item_batch)

q_entity = make_emb_posterior(all_entities)  # For sparse
q_entity_bias = make_bias_posterior(all_entities)
all_bias = q_entity_bias.sample(NB_SAMPLES)
all_feat = q_entity.sample(NB_SAMPLES)

# feat_users2 = tf.nn.embedding_lookup(all_feat, user_batch)
# feat_items2 = tf.nn.embedding_lookup(all_feat, item_batch)
# bias_users2 = tf.nn.embedding_lookup(all_bias, user_batch)
# bias_items2 = tf.nn.embedding_lookup(all_bias, item_batch)

# feat_users = emb_user_prior.sample()
# feat_items = emb_item_prior.sample()
# bias_users = bias_user_prior.sample(tf.shape(user_batch)[0])
# bias_items = bias_item_prior.sample()

N_SAMPLES = 10
if N_SAMPLES:
    feat_users = q_user.sample(N_SAMPLES)
    feat_items = q_item.sample(N_SAMPLES)
    bias_users = q_user_bias.sample(N_SAMPLES)
    bias_items = q_item_bias.sample(N_SAMPLES)
else:
    feat_users = q_user.sample()
    feat_items = q_item.sample()
    bias_users = q_user_bias.sample()
    bias_items = q_item_bias.sample()    
print(feat_users.shape)

mean_feat_users = q_user.mean()
mean_feat_items = q_item.mean()
mean_bias_users = q_user_bias.sample()
mean_bias_items = q_item_bias.sample()

# Predictions
def make_likelihood(feat_users, feat_items, bias_users, bias_items):
    logits = global_bias + tf.reduce_sum(feat_users * feat_items, 1) + bias_users + bias_items
    return tfd.Bernoulli(logits)

def make_likelihood_reg(sigma2, feat_users, feat_items, bias_users, bias_items):
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

def make_sparse_pred_reg(sigma2, x):
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
    return tfd.Normal(logits, scale=sigma2)

def define_variables(train_category, priors, sigma2, batch_size):
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
    uniq_user_rescale = tf.nn.embedding_lookup(priors, uniq_user_batch)[:, 0]
    uniq_item_rescale = tf.nn.embedding_lookup(priors, uniq_item_batch)[:, 0]
    entity_rescale = priors[:, 0]

    if is_classification:
        likelihood = make_likelihood(feat_users, feat_items, bias_users, bias_items)
        sparse_pred = make_sparse_pred(X_fm_batch)
    else:
        likelihood = make_likelihood_reg(sigma2, feat_users, feat_items, bias_users, bias_items)
        sparse_pred = make_sparse_pred_reg(sigma2, X_fm_batch)
    pred2 = sparse_pred.mean()
    # ll = make_likelihood(feat_users2, feat_items2, bias_users2, bias_items2)
    pred = tf.reduce_mean(likelihood.mean(), axis=0)  # À cause du nombre de samples
    pred_mean = global_bias + tf.reduce_sum(mean_feat_users * mean_feat_items, 1) + mean_bias_users + mean_bias_items
    pred = pred_mean

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
        #     nb_samples['train'] * likelihood.log_prob(outcomes) +
        #     # nb_samples['train'] * sparse_pred.log_prob(outcomes) +
        #     (nb_users + nb_items) / 2 * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
        #                                  emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users) +
        #                                  bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
        #                                  emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')

        # elbo = tf.reduce_mean(
        #     nb_samples[train_category] * likelihood.log_prob(outcomes) +
        #     nb_samples[train_category] * 1 / user_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
        #                       emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
        #     nb_samples[train_category] * 1 / item_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
        #                       emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')

        tmp = emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)

        print('l', likelihood)
        print('pred', pred.shape)

        print('sample unique', q_uniq_user.sample())
        print('sample', q_user.sample())
        print('2 samples unique', q_uniq_user.sample(2))
        print('2 samples', q_user.sample(2))

        print('l outcomes', likelihood.log_prob(outcomes).shape)
        print('kl', wtfd.kl_divergence(q_uniq_user, emb_user_prior).shape)
        print('all kl', wtfd.kl_divergence(q_uniq_user_bias, bias_user_prior) +
                            wtfd.kl_divergence(q_uniq_user, emb_user_prior) +
                            wtfd.kl_divergence(q_uniq_item_bias, bias_item_prior) +
                            wtfd.kl_divergence(q_uniq_item, emb_item_prior))
        print('mu_lambda', emb_user_mu0.log_prob(emb_user_mu).shape)

        ''' - tf.reduce_sum(wtfd.kl_divergence(q_uniq_user_bias, bias_user_prior) +
                            wtfd.kl_divergence(q_uniq_user, emb_user_prior)) +
            - tf.reduce_sum(wtfd.kl_divergence(q_uniq_item_bias, bias_item_prior) +
                            wtfd.kl_divergence(q_uniq_item, emb_item_prior))'''

        # Objective function ELBO
        elbo = (
            tf.reduce_mean(
                likelihood.log_prob(outcomes)
            ) -
            tf.reduce_mean(
                1 / uniq_user_rescale * (wtfd.kl_divergence(q_uniq_user_bias, bias_user_prior) +
                                    wtfd.kl_divergence(q_uniq_user, emb_user_prior))
            ) -
            tf.reduce_mean(
                1 / uniq_item_rescale * (wtfd.kl_divergence(q_uniq_item_bias, bias_item_prior) +
                                    wtfd.kl_divergence(q_uniq_item, emb_item_prior))
            ) + (
                global_bias_prior.log_prob(global_bias)
                + emb_user_mu0.log_prob(emb_user_mu) + tf.reduce_sum(precision_prior.log_prob(emb_user_lambda))
                + emb_item_mu0.log_prob(emb_item_mu) + tf.reduce_sum(precision_prior.log_prob(emb_item_lambda))
                + bias_user_mu0.log_prob(bias_user_mu) + precision_prior.log_prob(bias_user_lambda)
                + bias_item_mu0.log_prob(bias_item_mu) + precision_prior.log_prob(bias_item_lambda)
                + precision_prior.log_prob(alpha)
            ) / nb_samples['train']
        )   

        # many_feats = q_user.sample(100)

    sentinel = {
        'nb outcomes': tf.shape(outcomes),
        'nb samples': tf.constant(nb_samples[train_category]),
        'users': entity[:5, 0],
        'bias_user_mu': bias_user_mu,
        # 'user_batch': len(user_batch),
        # 'user unique batch': len(tf.unique(user_batch)),
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
        'likelihood (outcomes)': tf.shape(likelihood.log_prob(outcomes)),
        # 'pred': pred,
        'pred shape': tf.shape(pred),
        # 'mean pred': pred_mean,
        'mean pred shape': tf.shape(pred_mean),
        # 'pred2': pred2,
        'max pred': tf.reduce_max(pred),
        'min pred': tf.reduce_min(pred),
        # 'max pred2': tf.reduce_max(pred2),
        # 'min pred2': tf.reduce_min(pred2),
        # 'has nan': tf.reduce_any(tf.is_nan(pred2))
        # 'bias mean': bias_user_prior.mean(),
        # 'bias delta': bias_users[0] ** 2 / 2 + np.log(2 * np.pi) / 2,
        # 'sum bias delta': tf.reduce_sum(bias_users ** 2 / 2 + np.log(2 * np.pi) / 2)
    }

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

optimizer = tf.train.AdamOptimizer(gamma)  # 0.001
# optimizer = tf.train.GradientDescentOptimizer(gamma)  # 0.001
# optimizer = tf.train.MomentumOptimizer(gamma, momentum=0.9)  # 0.001
# optimizer = tf.train.RMSPropOptimizer(gamma)


def make_feed(category):
    return {user_batch: X[category][:, 0],
            item_batch: X[category][:, 1],
            outcomes: y[category]}#,
            # X_fm_batch: X_sp[category]}


class VFM:
    def __init__(self, train_category, valid_category, sigma2, stop_when_worse, train_patience=options.train_patience, valid_patience=options.valid_patience):
        self.start_time = time.time()
        self.data = {
            'train': train_category,
            'valid': valid_category,
            'test': 'test'
        }
        self.sigma2 = sigma2
        self.metrics = {
            'train': defaultdict(list),
            'valid': defaultdict(list),
            'test': defaultdict(list),
            'time': {}
        }
        self.category_watcher, self.metric_watcher = stop_when_worse
        self.train_patience = train_patience
        self.valid_patience = valid_patience
        print('START', self.model_name())

        self.batch_size = nb_samples[self.data['train']] // nb_iters  # All

    def model_name(self):
        if options.degenerate:
            title = 'fm-map'
        elif options.sparse:
            title = 'vfm-sparse'
        else:
            title = 'vfm'
        name = '{:s}-{:s}-{:s}-{:.2f}'.format(DATA, title, self.data['train'], self.sigma2)
        if SUFFIX:
            name += '-' + SUFFIX
        return name

    def stopping_rule(self):
        K = 5        
        if self.category_watcher == 'train':
            last_values = self.metrics[self.category_watcher][self.metric_watcher][-self.train_patience:]
            is_decreasing = self.metric_watcher in {'acc', 'auc', 'elbo'}  # If these metrics decrease, it's worse
            return (self.epoch >= MAX_EPOCHS or (len(last_values) >= self.train_patience and last_values == sorted(last_values, reverse=is_decreasing))), last_values
        elif self.epoch < K - 1 or self.epoch % self.valid_patience != 0:
            return False, []
        else:  # Try tracking quotient
            train = self.metrics['train']['elbo']
            strip = train[-K:]
            progress = 1000 * (sum(strip) / (K * max(strip)) - 1)
            valid = self.metrics['valid'][self.metric_watcher]  # Most possibly RMSE
            gen_loss = 100 * (valid[-1] / min(valid) - 1)
            quotient = gen_loss / progress
            return quotient > 0.2, valid[-2:]

    def save_metrics(self, category, epoch, y_truth, y_pred):
        if VERBOSE:
            print('[%s] pred' % category, y_truth[:5], y_pred[:5])
        if epoch not in self.metrics[category]['epoch']:
            self.metrics[category]['epoch'].append(epoch) 
        self.metrics[category]['acc'].append(np.mean(y_truth == np.round(y_pred)))
        self.metrics[category]['rmse'].append(mean_squared_error(y_truth, y_pred) ** 0.5)
        if set(y_truth) == {0., 1.}:
            self.metrics[category]['auc'].append(roc_auc_score(y_truth, y_pred))
            self.metrics[category]['nll'].append(log_loss(y_truth, y_pred, eps=1e-6))
        if VERBOSE:
            print('[%s] ' % category + ' '.join('{:s}={:f}'.format(metric, self.metrics[category][metric][-1]) for metric in self.metrics[category]))

    def run_and_save(self, category):
        valid_pred = sess.run(self.pred, feed_dict=make_feed(self.data[category]))
        self.save_metrics(category, self.epoch, y[self.data[category]], valid_pred)

    def save_logs(self):
        filename = '{:s}-{:d}.txt'.format(self.model_name(), int(round(time.time())))
        self.metrics['sigma2'] = self.sigma2
        self.metrics['model_name'] = self.model_name()
        # self.metrics['train']['elbo'] = self.metrics['train']['elbo'].astype()
        self.metrics['time']['total'] = time.time() - self.start_time
        with open('results/{:s}'.format(filename), 'w') as f:
            f.write(json.dumps({
                'description': DESCRIPTION,
                'date': datetime.now().isoformat(),
                'stopped': '{:d}/{:d}'.format(self.epoch, MAX_EPOCHS),
                'args': vars(options),
                'metrics': self.metrics,
            }, indent=4))

    def train(self):
        priors = tf.constant(nb_occurrences[self.data['train']][:, None].repeat(embedding_size, axis=1), dtype=np.float32)
        scale = tf.constant(self.sigma2)
        with tf.variable_scope(self.model_name()):
            infer_op, elbo, self.pred, likelihood, sentinel = define_variables(self.data['train'], priors, self.sigma2, self.batch_size)

        sess.run(tf.global_variables_initializer())
        
        self.epoch = 0
        while True:
            self.epoch += 1
            
            dt = time.time()
            train_elbos = []
            for nb_iter in range(nb_iters):
                batch_ids = np.random.randint(0, nb_samples[self.data['train']], size=self.batch_size)
                X['batch'] = X[self.data['train']][batch_ids]
                y['batch'] = y[self.data['train']][batch_ids]
                # X_sp['batch'] = make_sparse_tf(X_fm[i[self.data['train']]][batch_ids])

                _, train_elbo = sess.run([infer_op, elbo], feed_dict=make_feed('batch'))

                if VERBOSE >= 100:
                    values = sess.run([sentinel[key] for key in sentinel], feed_dict=make_feed('batch'))
                    for key, val in zip(sentinel, values):
                        print(key, val)
                    # sys.exit(0)

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

        for metric in self.metrics['test']:
            final = self.metrics['test'][metric][-1]
            best = (np.max if metric in {'auc', 'acc'} else np.min)(self.metrics['test'][metric])
            self.metrics['final ' + metric] = final
            self.metrics['best ' + metric] = float(best)
            print('[{:s}] final={:f} best={:f}'.format(metric, final, best))

        self.save_logs()

        # return self.metrics[self.category_watcher][self.metric_watcher][-self.train_patience]  # Best metric reported so far
        return min(self.metrics[self.category_watcher][self.metric_watcher][-2:])  # Min valid RMSE of last 2 steps


stopped_at = None
time_per_batch = None
time_per_epoch = None
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    train_writer = tf.summary.FileWriter('/tmp/test', sess.graph)
    
    if is_classification:
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
        best_sigma = sigma2s[np.argmin(valid_metrics)]

    refit = VFM('trainval', 'trainval', best_sigma, stop_when_worse=('train', 'elbo'))
    refit.train()

print('Finish', time.time() - start_time)
