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
        # batch_size = tf.shape(inputs)[0]  # Actually it is global now
        linear_in = tf.concat([inputs, h, tf.ones([batch_size, 1])], axis=1)
        linear_out = tf.matmul(linear_in, self._w)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return new_c, new_h


def bayesian_rnn(cell, inputs, y_i):
    initializer_c_h = (tf.zeros([batch_size, embedding_size]), tf.zeros([batch_size, embedding_size]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer_c_h)
    return h_list

    # If we're only interested in the last state
    # indices = tf.stack([seq_len, tf.range(batch_size)], axis=1)
    # relevant_outputs = tf.gather_nd(h_list, indices)
    # logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)

    # logits = tf.layers.dense(h_list, (nb_items * nb_classes))  # Not possible so we have to implement our own tensor product
    # Wo = tf.get_variable('weight', shape=(embedding_size, nb_items, nb_classes))
    # Well bias is missing, I'm lazy now, I just want the type check to shut up

    # logits = tf.tensordot(h_list, Wo, axes=1)  # shape = [max_seq_len, nb_batches, nb_items, nb_classes]
    # slicer = tf.one_hot(y_i, depth=nb_items)  # shape = [max_seq_len, nb_batches, nb_items (one-hot)]
    # relevant_logits = tf.einsum('ijkl,ijk->ijl', logits, slicer)  # shape = [max_seq_len, nb_batches, nb_classes]
    # return relevant_logits

# Data
embedding_size = 128
# None = 2
nb_classes = 5

# Training
batch_size = 2
nb_epochs = 10

nb_items = 10
items = {}
data = [
    [(1, 4), (2, 0), (3, 1)],  # , (3, 3)
    [(1, 3), (2, 1), (3, 0)],
    [(2, 0), (3, 1), (1, 2)],
    [(2, 1), (3, 0), (1, 1)]
]
nb_samples = len(data)
seq_len = len(data[0])
iters = nb_samples // batch_size
x_train = np.zeros((seq_len - 1, len(data), embedding_size))
y_i_train = []
y_v_train = []
for i_sample, sample in enumerate(data):
    for pos, (i, v) in enumerate(sample[:-1]):
        if (i, v) not in items:
            items[i, v] = np.random.random(embedding_size)
        x_train[pos, i_sample] = items[i, v]
    indices, values = zip(*sample[1:])
    y_i_train.append(indices)
    y_v_train.append(values)
y_i_train = np.array(y_i_train)
y_v_train = np.array(y_v_train)

# Loading data and config
# x_train = np.load('fraction.npy')
# print('Fraction data loaded', x_train.shape)
# 
# z_dim = 5

# Boilerplate
x = tf.placeholder(tf.float32, shape=[None, None, embedding_size], name='x')
y_i = tf.placeholder(tf.int32, shape=[None, None], name='y_i')
y_v = tf.placeholder(tf.int32, shape=[None, None], name='y_v')
seq_len = tf.shape(x)[0]

@zs.reuse('model')
def p_net(observed, seq_len):
    '''
    Decoder: p(x|z) = p(y_v|w)
    '''
    with zs.BayesianNet(observed=observed) as model:
        cell = BayesianLSTMCell(128, forget_bias=0.)
        # shape was [max_seq_len, nb_batches, nb_classes]
        h_list = bayesian_rnn(cell, x, y_i)
        item_features = tf.get_variable("item_features", shape=[nb_items, embedding_size, nb_classes],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
        relevant_items = tf.nn.embedding_lookup(item_features, y_i, name="feat_items")
        logits = tf.tensordot(h_list, relevant_items, axes=[[2], [2]])  # That's not even the good shape but anyway
        _ = zs.Categorical('y_v', logits)  # shape of its local_log_prob = [max_seq_len, nb_batches]
                                           # because we already observe the true variable (y_v is in observed)
    return model

def log_joint(observed):
    model = p_net(observed, seq_len)
    # print('all', model._stochastic_tensors)  # w and y_v
    log_pz, log_px_z = model.local_log_prob(['w', 'y_v'])  # Error
    # log_px_z = model.local_log_prob('y_v')
    return log_pz + log_px_z  # Error
    # return log_px_z

joint_ll = log_joint({'x': x, 'y_i': y_i, 'y_v': y_v})
cost = -joint_ll

# If I had all relevant_logits (shape: [max_seq_len, nb_batches, nb_classes]) I would do this:
# labels = tf.one_hot(y_v, depth=nb_classes)
# cost = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=labels,
#         logits=relevant_logits)

optimizer = tf.train.AdamOptimizer(0.001)
infer_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, nb_epochs + 1):
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            x_batch = x_train[:, t * batch_size:(t + 1) * batch_size]
            y_i_batch = y_i_train[t * batch_size:(t + 1) * batch_size]
            y_v_batch = y_v_train[t * batch_size:(t + 1) * batch_size]
            _, lb = sess.run([infer_op, cost],
                             feed_dict={x: x_batch, y_i: y_i_batch, y_v: y_v_batch})
            lbs.append(lb)

        print('Epoch {}: Lower bound = {}'.format(
              epoch, np.sum(lbs)))
