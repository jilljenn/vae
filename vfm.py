from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from scipy.sparse import coo_matrix, load_npz, save_npz
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
import pandas as pd
import numpy as np
import yaml
import json
import time
import sys


DESCRIPTION = 'Silent mode, not yet global_bias variational approximation'
start_time = time.time()

parser = argparse.ArgumentParser(description='Run VFM')
parser.add_argument('data', type=str, nargs='?', default='fraction')
parser.add_argument('--degenerate', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--sparse', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--regression', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--classification', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--patience', type=int, nargs='?', default=4)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--gamma', type=float, nargs='?', default=0.01)
parser.add_argument('--nb_batches', type=int, nargs='?', default=1)
options = parser.parse_args()


if getpass.getuser() == 'jj':
    PATH = '/home/jj'
else:
    PATH = '/Users/jilljenn/code'

DATA = options.data
print('Data is', DATA)
VERBOSE = 0

# Load data
if DATA in {'fraction', 'mangaki', 'movie1M', 'movie10M', 'movie100k'}:
    df = pd.read_csv(os.path.join(PATH, 'vae/data', DATA, 'data.csv'))
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
    if is_regression:
        y_fm = np.array(df['rating'])
    else:
        y_fm = np.array(df['outcome']).astype(np.float32)
    save_npz(X_fm_file, X_fm)
    np.save(os.path.join(PATH, 'vae/data', DATA, 'y_fm.npy'), y_fm)
else:
    X_fm = load_npz(X_fm_file)

i_trainval, i_test = train_test_split(list(range(nb_entries)), test_size=0.2)
i_train, i_valid = train_test_split(i_trainval, test_size=0.2)
data = {
    'train': df.iloc[i_train],
    'valid': df.iloc[i_valid],
    'trainval': df.iloc[i_trainval],
    'test': df.iloc[i_test]
}

X = {}
X_sp = {}
y = {}
nb_samples = {}
nb_occurrences = {
    'train': X_fm[i_train].sum(axis=0).A1,
    'trainval': X_fm[i_trainval].sum(axis=0).A1
}

for category in data:
    X[category] = np.array(data[category][['user', 'item']])
    print(category, X[category].size)
    if is_regression:
        y[category] = np.array(data[category]['rating']).astype(np.float32)
    else:
        y[category] = np.array(data[category]['outcome']).astype(np.float32)
    nb_samples[category] = len(X[category])
    indices = np.column_stack((np.arange(nb_samples[category]).repeat(2), X[category].flatten()))
    X_sp[category] = tf.SparseTensorValue(indices, np.ones(2 * nb_samples[category]), [nb_samples[category], nb_users + nb_items])

# Config
MAX_EPOCHS = 500
print('Nb samples', nb_samples['train'])
embedding_size = options.d
nb_iters = options.nb_batches
batch_size = nb_samples['train'] // nb_iters  # All
print('Nb iters', nb_iters)
gamma = options.gamma  # gamma 0.001 works better for classification

dt = time.time()

global_bias = tf.get_variable('global_bias', shape=[], initializer=tf.truncated_normal_initializer(stddev=0.1))
users = tf.get_variable('entities', shape=[nb_users + nb_items, 2 * embedding_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
bias = tf.get_variable('bias', shape=[nb_users + nb_items, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
all_entities = tf.constant(np.arange(nb_users + nb_items))

def make_mu():
    return tfd.Normal(loc=0., scale=1.)

def make_lambda():
    return tfd.Beta(1., 1.)

def make_embedding_prior():
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size, name='emb_prior')

def make_embedding_prior2(mu0, lambda0):
    return wtfd.MultivariateNormalDiag(loc=[mu0] * embedding_size, scale_diag=[1 / lambda0] * embedding_size)

def make_embedding_prior3(priors, entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch, name='priors_prec')
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=1 / tf.sqrt(prior_prec_entity), name='strong_emb_prior')

def make_bias_prior():
    # return tfd.Normal(loc=0., scale=1.)
    return wtfd.Normal(loc=0., scale=1., name='bias_prior')

def make_bias_prior2(mu0, lambda0):
    # return tfd.Normal(loc=0., scale=1.)
    return tfd.Normal(loc=mu0, scale=1/lambda0)

def make_bias_prior3(priors, entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch)
    return tfd.Normal(loc=0., scale=1/tf.sqrt(prior_prec_entity[:, 0]), name='strong_bias_prior')

def make_user_posterior(user_batch):
    feat_users = tf.nn.embedding_lookup(users, user_batch)
    # return tfd.Normal(loc=feat_users[:, :embedding_size], scale=feat_users[:, embedding_size:])
    if options.degenerate:
        std_devs = tf.zeros(embedding_size)
    else:
        # 1/tf.sqrt(prior_prec_entity)  # More precise if more ratings
        # tf.ones(embedding_size)  # Too imprecise
        std_devs = tf.nn.softplus(feat_users[:, :embedding_size])
    return wtfd.MultivariateNormalDiag(loc=feat_users[:, embedding_size:], scale_diag=std_devs, name='emb_posterior')

def make_entity_bias(entity_batch):
    bias_batch = tf.nn.embedding_lookup(bias, entity_batch)
    if options.degenerate:
        std_dev = 0.
    else:
        # 1/tf.sqrt(prior_prec_entity[:, 0])  # More precise if more ratings, should be clipped
        # 1.  # Too imprecise
        std_dev = tf.nn.softplus(bias_batch[:, 1])
    return tfd.Normal(loc=bias_batch[:, 0], scale=std_dev, name='bias_posterior')

# def make_item_posterior(item_batch):
#     items = tf.get_variable('items', shape=[nb_items, 2 * embedding_size])
#     feat_items = tf.nn.embedding_lookup(items, item_batch)
#     return tfd.Normal(loc=feat_items[:embedding_size], scale=feat_items[embedding_size:])

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_batch')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_batch')
X_fm_batch = tf.sparse_placeholder(tf.int32, shape=[None, nb_users + nb_items], name='sparse_batch')
outcomes = tf.placeholder(tf.float32, shape=[None], name='outcomes')

# mu0 = make_mu().sample()
# lambda0 = make_lambda().sample()

# emb_user_prior = make_embedding_prior2(mu0, lambda0)
# emb_item_prior = make_embedding_prior2(mu0, lambda0)
# bias_user_prior = make_bias_prior2(mu0, lambda0)
# bias_item_prior = make_bias_prior2(mu0, lambda0)

q_user = make_user_posterior(user_batch)
q_item = make_user_posterior(item_batch)
q_user_bias = make_entity_bias(user_batch)
q_item_bias = make_entity_bias(item_batch)

q_entity = make_user_posterior(all_entities)
q_entity_bias = make_entity_bias(all_entities)
all_bias = q_entity_bias.sample()
all_feat = q_entity.sample()
# feat_users2 = tf.nn.embedding_lookup(all_feat, user_batch)
# feat_items2 = tf.nn.embedding_lookup(all_feat, item_batch)
# bias_users2 = tf.nn.embedding_lookup(all_bias, user_batch)
# bias_items2 = tf.nn.embedding_lookup(all_bias, item_batch)

# feat_users = emb_user_prior.sample()
# feat_items = emb_item_prior.sample()
# bias_users = bias_user_prior.sample(tf.shape(user_batch)[0])
# bias_items = bias_item_prior.sample()

feat_users = q_user.sample()
feat_items = q_item.sample()
bias_users = q_user_bias.sample()
bias_items = q_item_bias.sample()

# Predictions
def make_likelihood(feat_users, feat_items, bias_users, bias_items):
    logits = global_bias + tf.reduce_sum(feat_users * feat_items, 1) + bias_users + bias_items
    return tfd.Bernoulli(logits)

def make_likelihood_reg(sigma2, feat_users, feat_items, bias_users, bias_items):
    logits = global_bias + tf.reduce_sum(feat_users * feat_items, 1) + bias_users + bias_items
    return tfd.Normal(logits, scale=sigma2, name='pred')

def make_sparse_pred(x):
    x = tf.cast(x, tf.float32)
    x2 = x ** 2
    w = tf.reshape(bias[:, 0], (-1, 1))  # Otherwise tf.matmul is crying
    V = users[:, embedding_size:]
    V2 = V ** 2
    logits = (tf.squeeze(tf.matmul(x, w, a_is_sparse=True)) +
              0.5 * tf.reduce_sum(tf.matmul(x, V, a_is_sparse=True) ** 2 -
                                  tf.matmul(x2, V2, a_is_sparse=True), axis=1))
    return tfd.Bernoulli(logits)

def make_sparse_pred_reg(sigma2, x):
    x = tf.cast(x, tf.float32)
    x2 = x# ** 2  # FIXME if x is 0/1 it's okay
    w = tf.reshape(all_bias, (-1, 1))
    # w = tf.reshape(bias[:, 0], (-1, 1))  # Otherwise tf.matmul is crying
    # V = users[:, embedding_size:]
    V = all_feat
    V2 = V ** 2
    logits = (tf.squeeze(tf.sparse_tensor_dense_matmul(x, w)) +
              0.5 * tf.reduce_sum(tf.sparse_tensor_dense_matmul(x, V) ** 2 -
                                  tf.sparse_tensor_dense_matmul(x2, V2), axis=1))
    return tfd.Normal(logits, scale=sigma2)

def define_variables(train_category, priors, sigma2):
    if options.degenerate:
        emb_user_prior = make_embedding_prior()
        emb_item_prior = make_embedding_prior()
        bias_user_prior = make_bias_prior()
        bias_item_prior = make_bias_prior()
    else:
        emb_user_prior = make_embedding_prior3(priors, user_batch)
        emb_item_prior = make_embedding_prior3(priors, item_batch)
        bias_user_prior = make_bias_prior3(priors, user_batch)
        bias_item_prior = make_bias_prior3(priors, item_batch)

    user_rescale = tf.nn.embedding_lookup(priors, user_batch)[:, 0]
    item_rescale = tf.nn.embedding_lookup(priors, item_batch)[:, 0]

    if is_classification:
        likelihood = make_likelihood(feat_users, feat_items, bias_users, bias_items)
    else:
        likelihood = make_likelihood_reg(sigma2, feat_users, feat_items, bias_users, bias_items)
    sparse_pred = make_sparse_pred_reg(sigma2, X_fm_batch)
    pred2 = sparse_pred.mean()
    # ll = make_likelihood(feat_users2, feat_items2, bias_users2, bias_items2)
    pred = likelihood.mean()
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
        elbo = tf.reduce_mean(
            nb_samples[train_category] * likelihood.log_prob(outcomes) +
            # nb_samples['train'] * sparse_pred.log_prob(outcomes) +
            (nb_users + nb_items) * 2 * (bias_user_prior.log_prob(bias_users) +
                                         emb_user_prior.log_prob(feat_users) +
                                         bias_item_prior.log_prob(bias_items) +
                                         emb_user_prior.log_prob(feat_items)), name='elbo')
    # / 2 : 1.27
    # * 2 : 1.16
    elif options.sparse:
        elbo = tf.reduce_mean(
            # nb_samples['train'] * likelihood.log_prob(outcomes) +
            nb_samples[train_category] * sparse_pred.log_prob(outcomes) +
            nb_samples[train_category] * 1/user_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
                              emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
            nb_samples[train_category] * 1/item_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
                              emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')
    else:
        # elbo = tf.reduce_mean(
        #     nb_samples['train'] * likelihood.log_prob(outcomes) +
        #     # nb_samples['train'] * sparse_pred.log_prob(outcomes) +
        #     (nb_users + nb_items) / 2 * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
        #                                  emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users) +
        #                                  bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
        #                                  emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')

        elbo = tf.reduce_mean(
            nb_samples[train_category] * likelihood.log_prob(outcomes) +
            nb_samples[train_category] * 1/user_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
                              emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
            nb_samples[train_category] * 1/item_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
                              emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)), name='elbo')

    sentinel = {
        'pred': pred[0],
        'outcome': outcomes[0],
        'll log prob': -likelihood.log_prob(outcomes),
        's ll log prob': -tf.reduce_sum(likelihood.log_prob(outcomes)),
        's pred delta': tf.reduce_sum((pred - outcomes) ** 2 / 2 + np.log(2 * np.pi) / 2),
        'bias sample': bias_users[0],
        'bias log prob': -bias_user_prior.log_prob(bias_users)[0],
        'sum bias log prob': -tf.reduce_sum(bias_user_prior.log_prob(bias_users)),
        'bias mean': bias_user_prior.mean(),
        'bias delta': bias_users[0] ** 2 / 2 + np.log(2 * np.pi) / 2,
        'sum bias delta': tf.reduce_sum(bias_users ** 2 / 2 + np.log(2 * np.pi) / 2)
    }

    infer_op = optimizer.minimize(-elbo)
    return infer_op, elbo, pred, pred2, likelihood, sentinel

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


def make_feed(category):
    return {user_batch: X[category][:, 0],
            item_batch: X[category][:, 1],
            outcomes: y[category],
            X_fm_batch: X_sp[category]}


class VFM:
    def __init__(self, train_category, valid_category, sigma2, stop_when_worse, patience=options.patience):
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
        self.patience = patience
        print('START', self.model_name())

    def model_name(self):
        if options.degenerate:
            title = 'fm-map'
        elif options.sparse:
            title = 'vfm-sparse'
        else:
            title = 'vfm'
        return '{:s}-{:s}-{:s}-{:.2f}'.format(DATA, title, self.data['train'], self.sigma2)

    def stopping_rule(self):
        last_values = self.metrics[self.category_watcher][self.metric_watcher][-self.patience:]
        is_decreasing = self.metric_watcher in {'acc', 'auc', 'elbo'}  # If these metrics decrease, it's worse
        return (self.epoch >= MAX_EPOCHS or (len(last_values) >= self.patience and last_values == sorted(last_values, reverse=is_decreasing))), last_values

    def save_metrics(self, category, epoch, y_truth, y_pred):
        if VERBOSE:
            print('[%s] pred' % category, y_truth[:5], y_pred[:5])
        self.metrics[category]['epoch'].append(epoch)
        self.metrics[category]['acc'].append(np.mean(y_truth == np.round(y_pred)))
        self.metrics[category]['rmse'].append(mean_squared_error(y_truth, y_pred) ** 0.5)
        if set(y_truth) == {0., 1.}:
            self.metrics[category]['auc'].append(roc_auc_score(y_truth, y_pred))
            self.metrics[category]['nll'].append(log_loss(y_truth, y_pred, eps=1e-6))
        if VERBOSE:
            print('[%s] ' % category + ' '.join('{:s}={:f}'.format(metric, self.metrics[category][metric][-1]) for metric in self.metrics[category]))

    def run_and_save(self, category):
        valid_pred, valid_pred2 = sess.run([self.pred, self.pred2], feed_dict=make_feed(self.data[category]))
        # if category == 'valid' and self.data['valid'] != 'valid':
        #     print(np.isnan(valid_pred).any())
        #     print(valid_pred)
        self.save_metrics(category, self.epoch, y[self.data[category]], valid_pred)

    def save_logs(self):
        filename = '{:s}-{:d}.txt'.format(self.model_name(), int(round(time.time())))
        self.metrics['sigma2'] = self.sigma2
        self.metrics['model_name'] = self.model_name()
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
            infer_op, elbo, self.pred, self.pred2, likelihood, sentinel = define_variables(self.data['train'], priors, self.sigma2)

        sess.run(tf.global_variables_initializer())
        
        self.epoch = 0
        while True:
            self.epoch += 1
            
            dt = time.time()
            train_elbos = []
            for nb_iter in range(nb_iters):
                batch_ids = np.random.randint(0, nb_samples[self.data['train']], size=batch_size)
                X['batch'] = X[self.data['train']][batch_ids]
                y['batch'] = y[self.data['train']][batch_ids]
                indices = np.column_stack((np.arange(batch_size).repeat(2), X['batch'].flatten()))
                X_sp['batch'] = tf.SparseTensorValue(indices, np.ones(2 * batch_size), [batch_size, nb_users + nb_items])

                _, train_elbo = sess.run([infer_op, elbo], feed_dict=make_feed('batch'))

                if VERBOSE >= 100:
                    values = sess.run([sentinel[key] for key in sentinel], feed_dict=make_feed('batch'))
                    for key, val in zip(sentinel, values):
                        print(key, val)

                train_elbos.append(train_elbo)
                if nb_iter == 0:
                    self.metrics['time']['per_batch'] = time.time() - dt
            if self.epoch == 1:
                self.metrics['time']['per_epoch'] = time.time() - dt

            if self.data['valid'] == 'valid' and self.epoch % 10 == 0:
                self.run_and_save('valid')
            has_to_stop, watched_values = self.stopping_rule()

            self.metrics['train']['epoch'].append(self.epoch)
            self.metrics['train']['elbo'].append(np.mean(train_elbos) / nb_samples[self.data['train']])
            if VERBOSE >= 10:
                self.run_and_save('train')
                
            if VERBOSE or self.epoch % 5000 == 0 or has_to_stop:
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

        return self.metrics[self.category_watcher][self.metric_watcher][-self.patience]  # Best metric reported so far


stopped_at = None
time_per_batch = None
time_per_epoch = None
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    train_writer = tf.summary.FileWriter('/tmp/test', sess.graph)
    
    if is_classification:
        best_sigma = 0.
    else:  # Have to find sigma2 via cross validation
        sigma2s = [0.01, 0.1, 0.2, 0.5, 1.]
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
