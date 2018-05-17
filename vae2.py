import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.contrib import layers
import numpy as np


def make_nn(x, out_size, hidden_size=(128, 64)):
    net = tf.layers.flatten(x)
    for size in hidden_size:
        net = tf.layers.dense(net, size, activation=tf.nn.relu)
    return tf.layers.dense(net, out_size)

def make_decoder(z, x_shape=(1, 20, 1)):
    '''
    Decoder: p(x|z)
    '''
    net = make_nn(z, 20)
    logits = tf.reshape(net, tf.concat([[-1], x_shape], axis=0))
    return tfd.Independent(tfd.Bernoulli(logits))

def make_encoder(x, z_dim=8):
    '''
    Encoder: q(z|x)
    '''
    net = make_nn(x, z_dim * 2)
    return tfd.MultivariateNormalDiag(loc=net[..., :z_dim],
                                      scale_diag=tf.nn.softplus(net[..., z_dim:]))

def make_prior(z_dim=8, dtype=tf.float32):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(z_dim, dtype))

# Loading data and config
x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
nb_samples, x_dim = x_train.shape
z_dim = 5
epochs = 1000
batch_size = 20
iters = nb_samples // batch_size

# Boilerplate
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
n = tf.shape(x)[0]

def log_joint(observed):
    model = p_net(observed, n, x_dim, z_dim)
    log_pz, log_px_z = model.local_log_prob(['z', 'x'])
    return log_pz + log_px_z

q_net = make_encoder(x)
z = q_net.sample()
p_net = make_decoder(z)
prior = make_prior()

print('p(x|z)', p_net.log_prob(x))
print('p(z)', prior.log_prob(z))
print('q(z|x)', q_net.log_prob(z))

lower_bound = tf.reduce_mean(
    p_net.log_prob(x) + prior.log_prob(z) - q_net.log_prob(z))

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(-lower_bound)  # Increase ELBO <=> Minimize -ELBO

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
            _, lb = sess.run([infer_op, lower_bound],
                             feed_dict={x: x_batch})
            lbs.append(lb)

        print('Epoch {}: Lower bound = {}'.format(
              epoch, np.sum(lbs)))
