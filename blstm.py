import tensorflow as tf
import zhusuan as zs
import numpy as np


class BayesianLSTMCell(object):
    def __init__(self, num_units, forget_bias=1.0):
        self._forget_bias = forget_bias
        w_mean = tf.zeros([2 * num_units + 1, 4 * num_units])
        self._w = zs.Normal('w', w_mean, std=1., group_ndims=2)

    def __call__(self, state, inputs):
        c, h = state
        print('state', state)
        # batch_size = tf.shape(inputs)[0]
        print('x', inputs)
        print('h', h)
        print('ones', tf.ones([batch_size, 1]))
        linear_in = tf.concat([inputs, h, tf.ones([batch_size, 1])], axis=1)
        linear_out = tf.matmul(linear_in, self._w)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return new_c, new_h


def bayesian_rnn(cell, inputs, seq_len):
    initializer_c_h = (tf.zeros([batch_size, embedding_size]), tf.zeros([batch_size, embedding_size]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer_c_h)
    seq_len_repeat = tf.tile([seq_len - 1], [batch_size])  # Normally, seq_len should be different for each batch
    indices = tf.stack([seq_len_repeat, tf.range(batch_size)], axis=1)
    relevant_outputs = tf.gather_nd(
        h_list, indices)
    # logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)
    logits = tf.squeeze(tf.layers.dense(h_list, 5), -1)
    return logits

# Data
embedding_size = 128
max_seq_len = 27

# Training
batch_size = 2
n_epochs = 10

nb_items = 10
items = {}
data = [
    [(1, 4), (2, 0), (3, 3)],
    [(1, 3), (2, 1)],
    [(2, 0)],
    [(2, 1), (3, 0)]
]
x_train = np.zeros((3, len(data), embedding_size))
for i_sample, sample in enumerate(data):
    for pos, (i, v) in enumerate(sample):
        if (i, v) not in items:
            items[i, v] = np.random.random(embedding_size)
        x_train[pos, i_sample] = items[i, v]


# Loading data and config
# x_train = np.load('fraction.npy')
# print('Fraction data loaded', x_train.shape)
# nb_samples, embedding_size = x_train.shape
# z_dim = 5
# epochs = 1000
# iters = nb_samples // batch_size

# Boilerplate
x = tf.placeholder(tf.float32, shape=[max_seq_len, None, embedding_size], name='x')
n = tf.shape(x)[0]

@zs.reuse('model')
def p_net(observed, n, x_dim, z_dim):
    '''
    Encoder: p(x|z)
    '''
    with zs.BayesianNet() as model:
        cell = BayesianLSTMCell(128, forget_bias=0.)
        logits = bayesian_rnn(cell, x, max_seq_len)
        _ = zs.Bernoulli('y', logits, dtype=tf.float32)
    return cell

def log_joint(observed):
    model = p_net(observed, n, x_dim, z_dim)
    log_pz, log_px_z = model.local_log_prob(['z', 'x'])
    return log_pz + log_px_z

# optimizer = tf.train.AdamOptimizer(0.001)
# infer_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[:, t * batch_size:(t + 1) * batch_size]
            _, lb = sess.run([infer_op, lower_bound],
                             feed_dict={x: x_batch})
            lbs.append(lb)

        print('Epoch {}: Lower bound = {}'.format(
              epoch, np.sum(lbs)))
