import tensorflow as tf
from tensorflow.contrib import layers
import zhusuan as zs
import numpy as np


@zs.reuse('model')
def p_net(observed, n, x_dim, z_dim):
    '''
    Encoder: p(x|z)
    '''
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1)
        lx_z = layers.fully_connected(z, 500)
        lx_z = layers.fully_connected(lx_z, 500)
        x_logits = layers.fully_connected(lx_z, x_dim,
                                          activation_fn=None)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model

@zs.reuse('variational')
def q_net(x, z_dim):
    '''
    Decoder: q(z|x)
    '''
    with zs.BayesianNet() as variational:
        lz_x = layers.fully_connected(tf.to_float(x), 500)
        lz_x = layers.fully_connected(lz_x, 500)
        z_mean = layers.fully_connected(lz_x, z_dim,
                                        activation_fn=None)
        z_logstd = layers.fully_connected(lz_x, z_dim,
                                          activation_fn=None)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1)
    return variational

# Loading data and config
x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
nb_samples, x_dim = x_train.shape
z_dim = 5
epochs = 1000
batch_size = 20
iters = nb_samples // batch_size

# Boilerplate
x = tf.placeholder(tf.int32, shape=[None, x_dim], name='x')
n = tf.shape(x)[0]

def log_joint(observed):
    model = p_net(observed, n, x_dim, z_dim)
    log_pz, log_px_z = model.local_log_prob(['z', 'x'])
    return log_pz + log_px_z

variational = q_net(x, z_dim)
qz_samples, log_qz = variational.query('z', outputs=True,
                                       local_log_prob=True)
lower_bound = zs.variational.elbo(
    log_joint, observed={'x': x}, latent={'z': [qz_samples, log_qz]})
cost = tf.reduce_mean(lower_bound.sgvb())
lower_bound = tf.reduce_sum(lower_bound)

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(cost)

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
