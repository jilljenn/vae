import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow.contrib.distributions as tfd
from tensorflow.contrib import layers
import numpy as np
from math import log
import sys


x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
nb_samples, x_dim = x_train.shape


epochs = 1000
Z_DIM = 2
batch_size = 20
nb_z_samples = 1
learning_rate = 0.001


def make_nn(x, out_size, hidden_size=(5,)):  # 128, 64
    net = x
    # net = tf.layers.flatten(x)
    for size in hidden_size:
        net = tf.layers.dense(net, size, activation=tf.nn.relu)
    return tf.layers.dense(net, out_size)

def make_decoder(z, x_shape=(x_dim,)):
    '''
    Decoder: p(x|z)
    '''
    with tf.variable_scope("decoder"):
        net = make_nn(z, x_dim)
        print('decoder net', net)
        logits = tf.reshape(net, tf.concat([[nb_z_samples, -1], x_shape], axis=0))  # For the batch
        print('logits', logits)
        return tfd.Independent(tfd.Bernoulli(logits), reinterpreted_batch_ndims=1)

def make_encoder(x, z_dim=Z_DIM):
    '''
    Encoder: q(z|x)
    '''
    with tf.variable_scope("encoder"):
        net = make_nn(x, z_dim * 2)
        print('encoder net', net)
        return tfd.MultivariateNormalDiag(loc=net[..., :z_dim],
                                          scale_diag=tf.nn.softplus(net[..., z_dim:]))

def make_prior(z_dim=Z_DIM, dtype=tf.float32):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(z_dim, dtype),
                                      scale_diag=tf.ones(z_dim, dtype))

# Loading data and config
iters = nb_samples // batch_size

# Boilerplate
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
# n = tf.shape(x)[0]

# def log_joint(observed):
#     model = p_net(observed, n, x_dim, z_dim)
#     log_pz, log_px_z = model.local_log_prob(['z', 'x'])
#     return log_pz + log_px_z

q_net = make_encoder(x)
z = q_net.sample(nb_z_samples)  # Just one sample of code
print('z shape', z.shape)
# sys.exit(0)
p_net = make_decoder(z)
prior = make_prior()

print('decoder', p_net)
print('mean decoder', p_net.mean())
print('obs', x)
# sys.exit(0)
# x is (batch_size, x_dim)
# z is (z_samples, batch_size, z_dim)
# encoder should be (batch: batch_size, event: z_dim)
# decoder should be (batch: (z_samples, batch_size), event: x_dim)
print('p(x|z)', p_net.log_prob(x))  # Should be (z_samples, batch_size)
print('p(z)', prior.log_prob(z))  # Should be (z_samples, batch_size)
print('q(z|x)', q_net.log_prob(z))  # Should be (z_samples, batch_size)
# sys.exit(0)

ll_output = p_net.log_prob(x)
print('ll output', ll_output.shape)
output = tf.reduce_mean(p_net.mean(), axis=0)
print('output', output.shape)

lower_bound = tf.reduce_mean(
    p_net.log_prob(x) + prior.log_prob(z) - q_net.log_prob(z))

optimizer = tf.train.AdamOptimizer(learning_rate)
infer_op = optimizer.minimize(-lower_bound)  # Increase ELBO <=> Minimize -ELBO

print("Nombre d'échantillons", nb_samples)
nb_parameters = int(np.sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('decoder')]) + nb_samples * Z_DIM)
nb_parameters_decoder = 0.#np.sum([np.prod(v.shape) for v in p_net.count_params()])
print("Nombre de paramètres", nb_parameters, nb_parameters_decoder)
print([(v, v.shape) for v in tf.trainable_variables() if v.name.startswith('decoder')])
input()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
            _, lb, o, ll_o = sess.run([infer_op, lower_bound, output, ll_output],
                             feed_dict={x: x_batch})
            # print('elbo', lb)
            lbs.append(lb)
            # print(o.shape, ll_o.shape, ll_o.sum())
            # print(ll_o)
            # print('owi', log_loss(x_batch.flatten(), o.flatten(), normalize=False))
            # sys.exit(0)

        o, ll = sess.run([output, ll_output], feed_dict={x: x_train})
        real_ll = -log_loss(x_train.flatten(), o.flatten(), normalize=False)
        # ll = real_ll = 0.

        # ll.sum() / nb_z_samples is also LL
        
        bic = log(nb_samples * x_dim) * nb_parameters - 2 * real_ll
        
        print('Epoch {}: ELBO = {:.3f}, LL = {:.3f}, BIC = {:.3f}'.format(
              epoch, nb_samples * np.mean(lbs), real_ll, bic))  # .sum()
        # sys.exit(0)

print(o.flatten())
print(x_train.flatten())
print(nb_samples * x_dim)
print(len(x_train.flatten()))
# 3200 pour batch, dim 8
