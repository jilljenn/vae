from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow.contrib.distributions as wtfd
from collections import Counter
import os.path
import pandas as pd
import numpy as np
import yaml
import sys


PATH = '/home/jj'
# PATH = '/Users/jilljenn/code'
DATA = 'movie'
VERBOSE = True

if DATA != 'movie':
    ratings = np.load('fraction.npy')
    print('Fraction data loaded')

    nb_users, nb_items = ratings.shape
    print(nb_users, 'users', nb_items, 'items')
    entries = []
    for i in range(nb_users):
        for j in range(nb_items):
            entries.append([i, nb_users + j, ratings[i][j]])  # FM format
    df = pd.DataFrame(entries, columns=('user', 'item', 'outcome'))
    # X_train = np.array(X_train)
else:
    with open(os.path.join(PATH, 'vae/data/movie100k/config.yml')) as f:
        config = yaml.load(f)
        nb_users = config['nb_users']
        nb_items = config['nb_items']
    df = pd.read_csv(os.path.join(PATH, 'vae/data/movie100k/data.csv'))
    df['item'] += nb_users  # FM format
    # X = np.array(df)

train, test = train_test_split(df, test_size=0.2)
print(train.head())
np_priors = np.zeros(nb_users + nb_items)
for k, v in Counter(train['user']).items():
    np_priors[k] = v
for k, v in Counter(train['item']).items():
    np_priors[k] = v
print('minimax', np_priors.min(), np_priors.max())
print(np_priors[nb_users - 5:nb_users + 5])

X_train = np.array(train)
X_test = np.array(test)

nb_samples, _ = X_train.shape

print('Nb samples', nb_samples)
embedding_size = 20
# batch_size = 5
batch_size = 128
# batch_size = nb_samples  # All
iters = nb_samples // batch_size
print('Nb iters', iters)
epochs = 200
gamma = 0.1


# tf.enable_eager_execution()  # Debug

users = tf.get_variable('entities', shape=[nb_users + nb_items, 2 * embedding_size])
bias = tf.get_variable('bias', shape=[nb_users + nb_items, 2])
priors = tf.constant(np_priors[:, None].repeat(embedding_size, axis=1), dtype=np.float32)

def make_mu():
    return tfd.Normal(loc=0., scale=1.)

def make_lambda():
    return tfd.Beta(1., 1.)

def make_embedding_prior():
    # return tfd.Normal(loc=[0.] * embedding_size, scale=[1.] * embedding_size)
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size)

def make_embedding_prior2(mu0, lambda0):
    return wtfd.MultivariateNormalDiag(loc=[mu0] * embedding_size, scale_diag=[1/lambda0] * embedding_size)

def make_embedding_prior3(entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch)
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=1000/prior_prec_entity)

def make_bias_prior():
    # return tfd.Normal(loc=0., scale=1.)
    return wtfd.Normal(loc=0., scale=1.)

def make_bias_prior2(mu0, lambda0):
    # return tfd.Normal(loc=0., scale=1.)
    return tfd.Normal(loc=mu0, scale=1/lambda0)

def make_bias_prior3(entity_batch):
    prior_prec_entity = tf.nn.embedding_lookup(priors, entity_batch)
    return tfd.Normal(loc=0., scale=1000/prior_prec_entity[:, 0])

def make_user_posterior(user_batch):
    feat_users = tf.nn.embedding_lookup(users, user_batch)
    # print('feat', feat_users)
    # return tfd.Normal(loc=feat_users[:, :embedding_size], scale=feat_users[:, embedding_size:])
    return wtfd.MultivariateNormalDiag(loc=feat_users[:, embedding_size:], scale_diag=tf.nn.softplus(feat_users[:, :embedding_size]))

def make_entity_bias(entity_batch):
    bias_batch = tf.nn.embedding_lookup(bias, entity_batch)
    # return tfd.Normal(loc=bias_batch[:, 0], scale=bias_batch[:, 1])
    return wtfd.Normal(loc=bias_batch[:, 0], scale=tf.nn.softplus(bias_batch[:, 1]))

# def make_item_posterior(item_batch):
#     items = tf.get_variable('items', shape=[nb_items, 2 * embedding_size])
#     feat_items = tf.nn.embedding_lookup(items, item_batch)
#     return tfd.Normal(loc=feat_items[:embedding_size], scale=feat_items[embedding_size:])

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_batch')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_batch')
outcomes = tf.placeholder(tf.int32, shape=[None], name='outcomes')

mu0 = make_mu().sample()
lambda0 = make_lambda().sample()

all_entities = tf.constant(np.arange(nb_users + nb_items))

# emb_user_prior = make_embedding_prior2(mu0, lambda0)
# emb_item_prior = make_embedding_prior2(mu0, lambda0)
# emb_user_prior = make_embedding_prior3(all_entities)
# emb_item_prior = make_embedding_prior3(all_entities)
emb_user_prior = make_embedding_prior()
emb_item_prior = make_embedding_prior()
# bias_user_prior = make_bias_prior2(mu0, lambda0)
# bias_item_prior = make_bias_prior2(mu0, lambda0)
# bias_user_prior = make_bias_prior3(all_entities)
# bias_item_prior = make_bias_prior3(all_entities)
bias_user_prior = make_bias_prior()
bias_item_prior = make_bias_prior()

q_user = make_user_posterior(user_batch)
q_item = make_user_posterior(item_batch)
q_user_bias = make_entity_bias(user_batch)
q_item_bias = make_entity_bias(item_batch)

q_entity = make_user_posterior(all_entities)
q_entity_bias = make_entity_bias(all_entities)
all_bias = q_entity_bias.sample()
all_feat = q_entity.sample()
feat_users2 = tf.nn.embedding_lookup(all_feat, user_batch)
feat_items2 = tf.nn.embedding_lookup(all_feat, item_batch)
bias_users2 = tf.nn.embedding_lookup(all_bias, user_batch)
bias_items2 = tf.nn.embedding_lookup(all_bias, item_batch)

feat_users = q_user.sample()
print('sample feat users', feat_users)
feat_items = q_item.sample()
bias_users = q_user_bias.sample()
bias_items = q_item_bias.sample()
user_rescale = tf.nn.embedding_lookup(priors, user_batch)[:, 0]
item_rescale = tf.nn.embedding_lookup(priors, item_batch)[:, 0]
# print(prior.cdf(1.7))
# for _ in range(3):
#     print(prior.sample([2]))

def make_likelihood(feat_users, feat_items, bias_users, bias_items):
    logits = tf.reduce_sum(feat_users * feat_items, 1) + bias_users + bias_items
    return tfd.Bernoulli(logits)

likelihood = make_likelihood(feat_users, feat_items, bias_users, bias_items)
ll = make_likelihood(feat_users2, feat_items2, bias_users2, bias_items2)
pred = likelihood.mean()
# print(likelihood.log_prob([1, 0]))

print('likelihood', likelihood.log_prob(outcomes))
print('prior', emb_user_prior.log_prob(feat_users))
print('scaled prior', emb_user_prior.log_prob(feat_users) / user_rescale)

print('posterior', q_user.log_prob(feat_users))
print('bias prior', bias_user_prior.log_prob(bias_users))
print('bias posterior', q_user_bias.log_prob(bias_users))

# sys.exit(0)

# sentinel = likelihood.log_prob(outcomes)
# sentinel = bias_prior.log_prob(bias_users)
sentinel = tf.reduce_sum(ll.log_prob(outcomes))
sentinel2 = tf.reduce_sum(likelihood.log_prob(outcomes))

print('tristesse', bias_user_prior.log_prob(all_bias))
print('desespoir', q_entity_bias.log_prob(all_bias))
print('stuff', emb_user_prior.log_prob(all_feat))

# sys.exit(0)

# elbo = tf.reduce_mean(
#     user_rescale * item_rescale * likelihood.log_prob(outcomes) +
#     item_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
#                     emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
#     user_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
#                     emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)))
elbo3 = tf.reduce_sum(
    likelihood.log_prob(outcomes) +
    1/user_rescale * (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
                      emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
    1/item_rescale * (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
                      emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)))

elbo = (len(X_train) * tf.reduce_mean(ll.log_prob(outcomes)) +
        tf.reduce_sum(bias_user_prior.log_prob(all_bias) - q_entity_bias.log_prob(all_bias)) +
        tf.reduce_sum(emb_user_prior.log_prob(all_feat) - q_entity.log_prob(all_feat)))

elbo2 = tf.reduce_mean(
    len(X_train) * likelihood.log_prob(outcomes) +
                     (bias_user_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
                      emb_user_prior.log_prob(feat_users) - q_user.log_prob(feat_users)) +
                     (bias_item_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
                      emb_user_prior.log_prob(feat_items) - q_item.log_prob(feat_items)))

# sys.exit(0)

optimizer = tf.train.AdamOptimizer(gamma)  # 0.001
# optimizer = tf.train.GradientDescentOptimizer(gamma)
infer_op = optimizer.minimize(-elbo)

y_train = X_train[:, 2]
y_test = X_test[:, 2]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(X_train)
        lbs = []
        for t in range(iters):
            X_batch = X_train[t * batch_size:(t + 1) * batch_size]
            _, lb, lb2, lb3, test, test2 = sess.run([infer_op, elbo, elbo2, elbo3, sentinel, sentinel2],
                             feed_dict={user_batch: X_batch[:, 0],
                                        item_batch: X_batch[:, 1],
                                        outcomes: X_batch[:, 2]})
            lbs.append(lb)

        train_pred = sess.run(pred, feed_dict={user_batch: X_train[:, 0],
                                               item_batch: X_train[:, 1]})
        if VERBOSE:
            print('Train ACC', np.mean(y_train == np.round(train_pred)))
            print('Train AUC', roc_auc_score(y_train, train_pred))
            print('Train NLL', log_loss(y_train, train_pred, eps=1e-6))
            print('Pred', y_train[:5], train_pred[:5])

        test_pred = sess.run(pred, feed_dict={user_batch: X_test[:, 0],
                                              item_batch: X_test[:, 1]})
        if VERBOSE:
            print('Test ACC', np.mean(y_test == np.round(test_pred)))
            print('Test AUC', roc_auc_score(y_test, test_pred))
            print('Test NLL', log_loss(y_test, test_pred, eps=1e-6))
            print('Pred', y_test[:5], test_pred[:5])

        print('Epoch {}: Lower bound = {} Other = {} Other = {}'.format(
              epoch, np.mean(lbs), lb2, lb3))
        # print('Sentinels', test, test2)
        # print(users[-8:].eval())
