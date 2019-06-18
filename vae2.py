import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import log_loss
import numpy as np
tfd = tfp.distributions


EMBEDDING_SIZE = 4
BATCH_SIZE = 10


# Loading data and config
x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
nb_samples, x_dim = x_train.shape


def make_nn(x, out_size, hidden_size=(128, 64)):
    layers = ([tf.keras.layers.Flatten()] +
              [tf.keras.layers.Dense(size, activation=tf.nn.relu)
               for size in hidden_size] +
              [tf.keras.layers.Dense(out_size)])
    return tf.keras.Sequential(layers)(x)


def make_decoder(z, x_shape=(x_dim,)):
    '''
    Decoder: p(x|z)
    '''
    net = make_nn(z, x_dim)
    logits = tf.reshape(net, (-1,) + x_shape)
    return logits, tfd.Independent(tfd.Bernoulli(logits))


def make_encoder(x, z_dim=EMBEDDING_SIZE):
    '''
    Encoder: q(z|x)
    '''
    net = make_nn(x, z_dim * 2)
    return tfd.MultivariateNormalDiag(
        loc=net[..., :z_dim],
        scale_diag=tf.nn.softplus(net[..., z_dim:]))


def make_prior(z_dim=EMBEDDING_SIZE, dtype=tf.float32):
    return tfd.MultivariateNormalDiag(loc=tf.zeros(z_dim, dtype))


epochs = 1000
iters = nb_samples // BATCH_SIZE

# Boilerplate
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
n = tf.shape(x)[0]

q_net = make_encoder(x)
z = q_net.sample()
logits, p_net = make_decoder(z)
proba = tf.sigmoid(logits)
prior = make_prior()

lower_bound = tf.reduce_mean(
    p_net.log_prob(x) + prior.log_prob(z) - q_net.log_prob(z))
#   log p(x|z)          p(z)                q(z|x)
sum_ll = tf.reduce_sum(p_net.log_prob(x))
# cross_entropy = tf.reduce_sum(
#     tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits))

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(-lower_bound)  # Increase ELBO <=> Minimize -ELBO


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[t * BATCH_SIZE:(t + 1) * BATCH_SIZE]
            _, lb = sess.run([infer_op, lower_bound],
                             feed_dict={x: x_batch})
            lbs.append(len(x_batch) * lb)

        train_proba, train_logits, train_sum_ll = sess.run(
            [proba, logits, sum_ll], feed_dict={x: x_train})
        print('Epoch {}: elbo={:.3f} (≤ sum_ll={:.3f})'.format(
              epoch, np.sum(lbs), train_sum_ll))
