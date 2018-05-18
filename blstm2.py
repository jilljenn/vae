# coding=utf8
from metrics import fetch_relevant_mean, compute_metrics, compute_binary_metrics, fetch_relevant_sum_per_batch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow.contrib.distributions as tfd
from tensorflow import distributions as tfd2
from time import time
import tensorflow as tf
import random
import numpy as np
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prior():
    # w_mean = tf.zeros([2 * embedding_size + 1, 4 * embedding_size])
    # return tfd.MultivariateNormalDiag(loc=w_mean,
    #                                   scale_diag=tf.ones(4 * embedding_size))
    return tfd.MultivariateNormalDiag(loc=tf.zeros(embedding_size))

class BayesianLSTMCell(object):
    def __init__(self, input_size, embedding_size, forget_bias, weights=None):
        self.embedding_size = embedding_size
        self._forget_bias = forget_bias
        # self._w = weights
        self._w = tf.get_variable(
            'w_mean', shape=[input_size + embedding_size + 1, 4 * embedding_size],
            initializer=tf.constant_initializer(0.))

    def __call__(self, state, inputs):
        c, h = state
        max_seq_len = tf.shape(inputs)[0]
        print('bs', batch_size)
        print('inputs', inputs)
        print('h', h)
        print('tf ones', tf.ones([batch_size, 1]))
        # inputs: [batch_size, input_size]
        # h: [batch_size, embedding_size]
        # ones: [batch_size, 1]
        linear_in = tf.concat([inputs, h, tf.ones([max_seq_len, 1])], axis=1)
        print('linear in', linear_in)
        linear_out = tf.matmul(linear_in, self._w)
        print('linear out', linear_out)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return new_c, new_h


def bayesian_rnn(cell, inputs, seq_len=None, y_i=None):
    batch_size = tf.shape(inputs)[1]
    print('encore batch size', batch_size)
    initializer_c_h = (tf.zeros([batch_size, cell.embedding_size]), tf.zeros([batch_size, cell.embedding_size]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer_c_h)
    print('h_list', h_list)

    if seq_len is not None:
        # If we're only interested in the last state
        indices = tf.stack([seq_len - 1, tf.range(batch_size)], axis=1)
        print('indices', indices)
        last_h_per_batch = tf.gather_nd(h_list, indices)
        return last_h_per_batch
        # logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)

    # We want all states (y_i is not None)

    # logits = tf.layers.dense(h_list, (nb_items * nb_classes))  # Not possible so we have to implement our own tensor product
    Wo = tf.get_variable('weight', shape=(embedding_size, nb_items))  # , nb_classes
    bias = tf.get_variable('bias', shape=(nb_items,))  # , nb_classes

    logits = tf.tensordot(h_list, Wo, axes=1) + bias  # shape = [max_seq_len, nb_batches, nb_items, nb_classes]
    slicer = tf.one_hot(y_i, depth=nb_items)  # shape = [max_seq_len, nb_batches, nb_items (one-hot)]
    relevant_logits = tf.einsum('ijk,ijk->ij', logits, slicer)  # shape = [max_seq_len, nb_batches, nb_classes]
    print('this should be S, B', relevant_logits)
    return relevant_logits

# Data
# embedding_size = 128
input_size = 128
embedding_size = 32
# None = 2
# nb_classes = 10

# Training
batch_size = 50
nb_epochs = 20
nb_mc_samples = 5

# nb_items = 10
items = {}

# Dummy
# data = [
#     [(1, 4), (2, 0), (3, 1), (3, 3)],
#     [(1, 3), (2, 1), (3, 0)],
#     [(2, 0), (3, 1)],
#     [(2, 1), (3, 0), (1, 1)]
# ]
# nb_items = 5

# DATA = 'assist09'
DATA = 'ml0-3'
with open('{:s}.pickle'.format(DATA), 'rb') as f:
    data = pickle.load(f)
    nb_items = pickle.load(f)

nb_users = len(data)
seq_lengths = np.array(list(map(len, data))) - 1
max_seq_len = max(seq_lengths)

x_all = np.zeros((max_seq_len, len(data), input_size))
y_i_all = np.zeros((max_seq_len, len(data)))
y_v_all = np.zeros((max_seq_len, len(data)))
for i_sample, sample in enumerate(data):
    for pos, (i, v) in enumerate(sample[:-1]):
        if (i, v) not in items:
            items[i, v] = np.random.multivariate_normal(np.zeros(input_size), np.eye(input_size))
        x_all[pos, i_sample] = items[i, v]
    indices, values = zip(*sample[1:])
    # print(indices, values, seq_lengths[i_sample], y_i_train.shape, y_v_train.shape)
    y_i_all[:seq_lengths[i_sample], i_sample] = indices
    y_v_all[:seq_lengths[i_sample], i_sample] = values
y_i_all = np.array(y_i_all)
y_v_all = np.array(y_v_all)
print('Data prepared')

i_train, i_test = train_test_split(range(nb_users), test_size=0.2)
nb_train_users = len(i_train)

x_train = x_all[:, i_train, :]
y_i_train = y_i_all[:, i_train]
y_v_train = y_v_all[:, i_train]
seq_lengths_train = seq_lengths[i_train]

x_test = x_all[:, i_test, :]
y_i_test = y_i_all[:, i_test]
y_v_test = y_v_all[:, i_test]
seq_lengths_test = seq_lengths[i_test]

iters = nb_train_users // batch_size

# Loading data and config
# x_train = np.load('fraction.npy')
# print('Fraction data loaded', x_train.shape)
# 
# z_dim = 5

# Boilerplate
# w = tf.placeholder(tf.float32, shape=[2 * embedding_size + 1, 4 * embedding_size], name='w')
x = tf.placeholder(tf.float32, shape=[None, None, input_size], name='x')
y_i = tf.placeholder(tf.int32, shape=[None, None], name='y_i')
y_v = tf.placeholder(tf.int32, shape=[None, None], name='y_v')
seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')

def make_dec(z, y_i):
    item_features = tf.get_variable("item_features", shape=[nb_items, embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
    relevant_items = tf.nn.embedding_lookup(item_features, y_i, name="feat_items")
    print('z', z)
    print('relevant', relevant_items)
    logits = tf.einsum('jk,ijk->ij', z, relevant_items)
    return tfd2.Bernoulli(logits), logits

def make_decoder(y_i):
    '''
    Decoder: p(x|z) = p(y_v|w)
    '''
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        cell = BayesianLSTMCell(input_size, embedding_size, forget_bias=0.)
        # shape was [max_seq_len, nb_batches, nb_classes]
        relevant_logits = bayesian_rnn(cell, x, y_i=y_i)
        print('relevant', relevant_logits)
        
        # print('hl', h_list)
        # print('ri', relevant_items)
        # logits = tf.tensordot(h_list, relevant_items, axes=[[2], [2]])  # That's not even the good shape but anyway
        # y = tfd.Categorical(logits=relevant_logits)  # shape of its local_log_prob = [max_seq_len, nb_batches]
                                           # because we already observe the true variable (y_v is in observed)
        # y = tfd.Independent(tfd.Bernoulli(logits=relevant_logits))
        y = tfd2.Bernoulli(relevant_logits)
        print('y', y)
        return y, relevant_logits

# def make_encoder():
#     '''
#     Encoder: q(z|x) = q(w|y_v)
#     Normal distribution
#     '''
#     w_mean = tf.get_variable(
#         'w_mean', shape=[embedding_size * 2 + 1, 4 * embedding_size],
#         initializer=tf.constant_initializer(0.))
#     w_logstd = tf.get_variable(
#         'w_logstd', shape=[embedding_size * 2 + 1, 4 * embedding_size],
#         initializer=tf.constant_initializer(-3.))
#     return tfd2.Normal(loc=w_mean, scale=tf.nn.softplus(w_logstd))

def make_enc(x):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        cell = BayesianLSTMCell(input_size, 2 * embedding_size, forget_bias=0.)
        relevant_logits = bayesian_rnn(cell, x, seq_len=seq_len)
        print('size of relevant logits', relevant_logits)
        return tfd.MultivariateNormalDiag(loc=relevant_logits[..., :embedding_size],
                                          scale_diag=tf.nn.softplus(relevant_logits[..., embedding_size:]))
        # return tfd2.Normal(loc=relevant_logits[..., :embedding_size],
        #                    scale=tf.nn.softplus(relevant_logits[..., embedding_size:]))

def compute_elbo():
    q_net = make_enc(x)
    z = q_net.sample()
    p_net, logits = make_dec(z, y_i)
    # p_net, logits = make_decoder(y_i)
    prior = make_prior()
    # ws = q_net.sample(nb_mc_samples)

    all_log_px_z = p_net.log_prob(y_v)
    # print('all log prob', all_log_px_z)
    log_px_z = fetch_relevant_sum_per_batch(all_log_px_z, seq_len)
    print('log_px_z', log_px_z)
    # print('log_pz', prior.log_prob(z))
    # print('log_qz_x', q_net.log_prob(z))
    elbo = log_px_z + prior.log_prob(z) - q_net.log_prob(z)
    print('elbo', elbo)
    return tf.reduce_mean(elbo), logits

    score = []
    for i_sample in range(nb_mc_samples):
        p_net, relevant_logits = make_decoder(ws[i_sample])
        # print('all', model._stochastic_tensors)  # w and y_v
        # log_pz, log_px_z = model.local_log_prob(['w', 'y_v'])  # Error
        # print('poids', cell._w)
        print('values', y_v)
        log_pz = tf.reduce_sum(prior.log_prob(ws[i_sample]))
        print('log prior', log_pz)
        # log_pz = 0
        log_px_z = fetch_relevant_mean(p_net.log_prob(y_v), seq_len, normalizer=1)  # Is it batch size?
        print('per batch', p_net.log_prob(y_v))
        print('log likelihood', log_px_z)
        # log_px_z = model.local_log_prob('y_v')
        log_qz_x = tf.reduce_sum(q_net.log_prob(ws[i_sample]))
        print('log posterior', log_qz_x)
        score.append(log_pz + log_px_z - log_qz_x)
        # score.append(log_px_z)

    return tf.reduce_mean(score), relevant_logits  # Error
    # return log_px_z

elbo, logits = compute_elbo()
# print('logits', logits)
cost = -elbo

# Useful values for metrics
mask = tf.transpose(tf.sequence_mask(seq_len, maxlen=tf.shape(y_v)[0]))
print('mask', mask)
all_pred = tf.boolean_mask(tf.sigmoid(logits), mask)
all_truth = tf.boolean_mask(y_v, mask)
macc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(all_truth, tf.float32), tf.round(all_pred)), tf.float32))

# If I had all relevant_logits (shape: [max_seq_len, nb_batches, nb_classes]) I would do this:
# labels = tf.one_hot(y_v, depth=nb_classes)
# cost = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=labels,
#         logits=relevant_logits)

optimizer = tf.train.AdamOptimizer(0.01)
infer_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_truth = sess.run(all_truth, feed_dict={y_v: y_v_train, seq_len: seq_lengths_train})
    test_truth = sess.run(all_truth, feed_dict={y_v: y_v_test, seq_len: seq_lengths_test})

    for epoch in range(1, nb_epochs + 1):
        start = time()
        np.random.shuffle(x_train)
        lbs = []
        for t in range(iters):
            seq_len_batch = seq_lengths_train[t * batch_size:(t + 1) * batch_size]
            max_seq_len_batch = max(seq_len_batch)

            x_batch = x_train[:max_seq_len_batch, t * batch_size:(t + 1) * batch_size]
            y_i_batch = y_i_train[:max_seq_len_batch, t * batch_size:(t + 1) * batch_size]
            y_v_batch = y_v_train[:max_seq_len_batch, t * batch_size:(t + 1) * batch_size]
            # print('seq', seq_len_batch)
            # print('x.shape', x_batch.shape)
            # print('yi', y_i_batch.shape)
            # print('yv', y_v_batch.shape)
            _, lb = sess.run([infer_op, elbo],
                             feed_dict={x: x_batch, y_i: y_i_batch, y_v: y_v_batch, seq_len: seq_len_batch})
            lbs.append(lb)

        lb_train, train_logits, train_macc, train_pred = sess.run([elbo, logits, macc, all_pred],
                         feed_dict={x: x_train, y_i: y_i_train, y_v: y_v_train, seq_len: seq_lengths_train})
        lb_test, test_logits, test_macc, test_pred = sess.run([elbo, logits, macc, all_pred],
                         feed_dict={x: x_test, y_i: y_i_test, y_v: y_v_test, seq_len: seq_lengths_test})
        # train_macc = compute_binary_metrics(y_v_train, train_logits, seq_lengths_train)
        # test_macc = compute_binary_metrics(y_v_test, test_logits, seq_lengths_test)
        # print(acc.eval()[:10])
        print('Epoch {}: (Train ELBO={} ACC={:f} AUC={:f}) (Test ELBO={} ACC={:f} AUC={:f}) {:.3f} s'.format(
              epoch, np.sum(lbs), train_macc, roc_auc_score(train_truth, train_pred),
                     lb_test, test_macc, roc_auc_score(test_truth, test_pred),
                     time() - start))

    lb, test_logits = sess.run([elbo, logits],
                                  feed_dict={x: x_test, y_i: y_i_test, y_v: y_v_test, seq_len: seq_lengths_test})
    # acc, test_rmse, test_macc = compute_metrics(y_v_test, test_logits, seq_lengths_test)
    # test_proba = tf.nn.softmax(test_logits)
    test_proba = tf.sigmoid(test_logits)
    # print('test proba', test_proba)
    # print('seq len', seq_lengths_test)
    # test_pred = tf.argmax(test_proba, 2)
    # print('Test error', lb)
    # print('snif', test_rmse.eval(), test_macc.eval())

    # Display pred for random test sample
    i = random.choice(range(x_test.shape[1]))
    pred = test_proba[:seq_lengths_test[i], i].eval()
    print(np.round(pred, 1))
    truth = y_v_test[:seq_lengths_test[i], i]
    print(truth)

    # print('Computing AUC…')
    # this_max_seq_len = tf.shape(test_logits)[0]
    
    
    # auc = roc_auc_score(all_truth.eval({x: x_test, y_i: y_i_test, y_v: y_v_test, seq_len: seq_lengths_test}), all_pred.eval())
    # print('AUC:', auc)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    test_writer = tf.summary.FileWriter('/tmp/test')
