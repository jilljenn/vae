from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow.contrib.distributions as wtfd
import pandas as pd
import numpy as np
import yaml
import sys


DATA = 'movie'

if DATA != 'movie':
    ratings = np.load('fraction.npy')
    print('Fraction data loaded')

    nb_users, nb_items = ratings.shape
    entries = []
    for i in range(nb_users):
        for j in range(nb_items):
            entries.append([i, nb_users + j, ratings[i][j]])  # FM format
    df = pd.DataFrame(entries)
    # X_train = np.array(X_train)
else:
    with open('/home/jj/vae/data/movie100k/config.yml') as f:
        config = yaml.load(f)
        nb_users = config['nb_users']
        nb_items = config['nb_items']
    df = pd.read_csv('/home/jj/vae/data/movie100k/data.csv')
    df['item'] += nb_users  # FM format
    # X = np.array(df)

train, test = train_test_split(df, test_size=0.2)
X_train = np.array(train)
X_test = np.array(test)

nb_samples, _ = X_train.shape

print('Nb samples', nb_samples)
embedding_size = 20
# batch_size = 5
# batch_size = 128
batch_size = nb_samples  # All
iters = nb_samples // batch_size
print('Nb iters', iters)
epochs = 200
gamma = 0.1


# tf.enable_eager_execution()  # Debug

def make_prior():
    # return tfd.Normal(loc=[0.] * embedding_size, scale=[1.] * embedding_size)
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size)

def make_mu():
    return tfd.Normal(loc=0., scale=1.)

def make_lambda():
    return tfd.Beta(1., 1.)

def make_embedding_prior2(mu0, lambda0):
    return wtfd.MultivariateNormalDiag(loc=[mu0] * embedding_size, scale_diag=[1/lambda0] * embedding_size)

def make_bias_prior():
    # return tfd.Normal(loc=0., scale=1.)
    return wtfd.Normal(loc=0., scale=1.)

def make_bias_prior2(mu0, lambda0):
    # return tfd.Normal(loc=0., scale=1.)
    return tfd.Normal(loc=mu0, scale=1/lambda0)

users = tf.get_variable('entities', shape=[nb_users + nb_items, 2 * embedding_size])
bias = tf.get_variable('bias', shape=[nb_users + nb_items, 2])

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

prior = make_embedding_prior2(mu0, lambda0)
bias_prior = make_bias_prior2(mu0, lambda0)
q_user = make_user_posterior(user_batch)
q_item = make_user_posterior(item_batch)
q_user_bias = make_entity_bias(user_batch)
q_item_bias = make_entity_bias(item_batch)
feat_users = q_user.sample()
print('sample feat users', feat_users)
feat_items = q_item.sample()
bias_users = q_user_bias.sample()
bias_items = q_item_bias.sample()
# print(prior.cdf(1.7))
# for _ in range(3):
#     print(prior.sample([2]))

def make_likelihood(feat_users, feat_items, bias_users, bias_items):
    logits = tf.reduce_sum(feat_users * feat_items, 1) + bias_users + bias_items
    return tfd.Bernoulli(logits)

likelihood = make_likelihood(feat_users, feat_items, bias_users, bias_items)
pred = likelihood.mean()
# print(likelihood.log_prob([1, 0]))

print('likelihood', likelihood.log_prob(outcomes))
print('prior', prior.log_prob(feat_users))
print('posterior', q_user.log_prob(feat_users))
print('bias prior', bias_prior.log_prob(bias_users))
print('bias posterior', q_user_bias.log_prob(bias_users))

# sentinel = likelihood.log_prob(outcomes)
# sentinel = bias_prior.log_prob(bias_users)
sentinel = q_user_bias.log_prob(bias_users)

elbo = tf.reduce_mean(
    len(X_train) * likelihood.log_prob(outcomes) +
    bias_prior.log_prob(bias_users) - q_user_bias.log_prob(bias_users) +
    bias_prior.log_prob(bias_items) - q_item_bias.log_prob(bias_items) +
    prior.log_prob(feat_users) - q_user.log_prob(feat_users) +
    prior.log_prob(feat_items) - q_item.log_prob(feat_items))

# sys.exit(0)

optimizer = tf.train.AdamOptimizer(gamma)  # 0.001
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
            _, lb, test = sess.run([infer_op, elbo, sentinel],
                             feed_dict={user_batch: X_batch[:, 0],
                                        item_batch: X_batch[:, 1],
                                        outcomes: X_batch[:, 2]})
            # print(test)
            lbs.append(lb)

        train_pred = sess.run(pred, feed_dict={user_batch: X_train[:, 0],
                                               item_batch: X_train[:, 1]})
        print('Train ACC', np.mean(y_train == np.round(train_pred)))
        print('Train AUC', roc_auc_score(y_train, train_pred))
        print('Train NLL', log_loss(y_train, train_pred))

        test_pred = sess.run(pred, feed_dict={user_batch: X_test[:, 0],
                                              item_batch: X_test[:, 1]})
        print('Test ACC', np.mean(y_test == np.round(test_pred)))
        print('Test AUC', roc_auc_score(y_test, test_pred))
        print('Test NLL', log_loss(y_test, test_pred))

        print('Epoch {}: Lower bound = {}'.format(
              epoch, np.mean(lbs) / len(X_train)))
        # print(users[-8:].eval())
