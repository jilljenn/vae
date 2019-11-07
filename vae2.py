from sklearn.metrics import roc_auc_score, log_loss
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from tensorflow.contrib import layers
import numpy as np
import sys


x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
nb_samples, x_dim = x_train.shape


Z_DIM = 8
batch_size = 30


def make_nn(x, out_size, hidden_size=(16,)):  # 128, 64
    net = tf.layers.flatten(x)
    for size in hidden_size:
        net = tf.layers.dense(net, size, activation=tf.nn.relu)
    return tf.layers.dense(net, out_size)

def make_decoder(z, x_shape=(x_dim,)):
    '''
    Decoder: p(x|z)
    '''
    with tf.variable_scope("decoder"):
        net = make_nn(z, x_dim)
        logits = tf.reshape(net, tf.concat([[1, -1], x_shape], axis=0))  # For the batch
        return tfd.Independent(tfd.Bernoulli(logits))

def make_encoder(x, z_dim=Z_DIM):
    '''
    Encoder: q(z|x)
    '''
    with tf.variable_scope("encoder"):
        net = make_nn(x, z_dim * 2)
        return tfd.MultivariateNormalDiag(loc=net[..., :z_dim],
                                          scale_diag=tf.nn.softplus(net[..., z_dim:]))

def make_prior(z_dim=Z_DIM, dtype=tf.float32):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(z_dim, dtype))

# Loading data and config
epochs = 1000
iters = nb_samples // batch_size

# Boilerplate
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')  # Does not like None
# n = tf.shape(x)[0]

# def log_joint(observed):
#     model = p_net(observed, n, x_dim, z_dim)
#     log_pz, log_px_z = model.local_log_prob(['z', 'x'])
#     return log_pz + log_px_z

q_net = make_encoder(x)
z = q_net.sample()  # Just one sample of code
print('shape', z.shape)
# sys.exit(0)
p_net = make_decoder(z)
prior = make_prior()

print('p(x|z)', p_net.log_prob(x))
print('p(z)', prior.log_prob(z))
print('q(z|x)', q_net.log_prob(z))

ll_output = p_net.log_prob(x)
print('ll output', ll_output.shape)
output = p_net.sample(1)
print('output', output.shape)

lower_bound = tf.reduce_mean(
    p_net.log_prob(x) + prior.log_prob(z) - q_net.log_prob(z))

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(-lower_bound)  # Increase ELBO <=> Minimize -ELBO

print("Nombre d'échantillons", nb_samples)
nb_parameters = np.sum([np.prod(v.shape) for v in tf.trainable_variables() if v.name.startswith('decoder')])
nb_parameters_decoder = 0.#np.sum([np.prod(v.shape) for v in p_net.count_params()])
print("Nombre de paramètres", nb_parameters, nb_parameters_decoder)
print([(v, v.shape) for v in tf.trainable_variables() if v.name.startswith('decoder')])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[t * batch_size:(t + 1) * batch_size]
            _, lb, o, ll_o = sess.run([infer_op, lower_bound, output, ll_output],
                             feed_dict={x: x_batch})
            print(lb.shape)
            lbs.append(lb)
            print(o.shape, ll_o.shape, ll_o.sum())
            print(ll_o)
            print('owi', log_loss(x_batch.flatten(), o.flatten(), normalize=False))
            sys.exit(0)

        # o, ll = sess.run([output, ll_output], feed_dict={x: x_train})
        # real_ll = log_loss(x_train.flatten(), o.flatten(), normalize=False)
        ll = real_ll = 0.
        print('Epoch {}: Lower bound = {}, LL = {}, also LL = {}'.format(
              epoch, np.sum(lbs), ll, real_ll))  # .sum()
