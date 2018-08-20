import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow.contrib.distributions as wtfd
import numpy as np


ratings = np.load('fraction.npy')
print('Fraction data loaded')

nb_users, nb_items = ratings.shape
x_train = []
for i in range(nb_users):
    for j in range(nb_items):
        x_train.append([i, j, ratings[i][j]])
x_train = np.array(x_train)
nb_samples, _ = x_train.shape
print('Nb samples', nb_samples)
embedding_size = 5
batch_size = 10
iters = nb_samples // batch_size
epochs = 100


# tf.enable_eager_execution()  # Debug


def make_prior():
    # return tfd.Normal(loc=[0.] * embedding_size, scale=[1.] * embedding_size)
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size)

users = tf.get_variable('entities', shape=[nb_users + nb_items, 2 * embedding_size])

def make_user_posterior(user_batch):
    feat_users = tf.nn.embedding_lookup(users, user_batch)
    # print('feat', feat_users)
    # return tfd.Normal(loc=feat_users[:, :embedding_size], scale=feat_users[:, embedding_size:])
    return wtfd.MultivariateNormalDiag(loc=feat_users[:, :embedding_size], scale_diag=feat_users[:, embedding_size:])

# def make_item_posterior(item_batch):
#     items = tf.get_variable('items', shape=[nb_items, 2 * embedding_size])
#     feat_items = tf.nn.embedding_lookup(items, item_batch)
#     return tfd.Normal(loc=feat_items[:embedding_size], scale=feat_items[embedding_size:])

user_batch = tf.placeholder(tf.int32, shape=[None], name='user_batch')
item_batch = tf.placeholder(tf.int32, shape=[None], name='item_batch')
outcomes = tf.placeholder(tf.int32, shape=[None], name='outcomes')

#with tf.Session() as sess:
prior = make_prior()
q_user = make_user_posterior(user_batch)
q_item = make_user_posterior(item_batch)
feat_users = q_user.sample()
print('sample feat users', feat_users)
feat_items = q_item.sample()
# print(prior.cdf(1.7))
# for _ in range(3):
#     print(prior.sample([2]))

def make_likelihood(feat_users, feat_items):
    logits = tf.reduce_sum(feat_users * feat_items, 1)
    return tfd.Bernoulli(logits)

likelihood = make_likelihood(feat_users, feat_items)
# print(likelihood.log_prob([1, 0]))

print('likelihood', likelihood.log_prob(outcomes))
print('prior', prior.log_prob(feat_users))
print('posterior', q_user.log_prob(feat_users))
elbo = tf.reduce_mean(
    likelihood.log_prob(outcomes) +
    prior.log_prob(feat_users) - q_user.log_prob(feat_users) +
    prior.log_prob(feat_items) - q_item.log_prob(feat_items))

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(-elbo)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
            _, lb = sess.run([infer_op, elbo],
                             feed_dict={user_batch: x_batch[:, 0],
                                        item_batch: x_batch[:, 1],
                                        outcomes: x_batch[:, 2]})
            lbs.append(lb)

        print('Epoch {}: Lower bound = {}'.format(
              epoch, np.sum(lbs)))
