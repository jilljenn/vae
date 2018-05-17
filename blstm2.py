from metrics import fetch_relevant_mean, compute_metrics
from sklearn.model_selection import train_test_split
import tensorflow.contrib.distributions as tfd
from tensorflow import distributions as tfd2
import tensorflow as tf
import numpy as np
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prior():
    w_mean = tf.zeros([2 * embedding_size + 1, 4 * embedding_size])
    return tfd.MultivariateNormalDiag(loc=w_mean,
                                      scale_diag=tf.ones(4 * embedding_size))

class BayesianLSTMCell(object):
    def __init__(self, embedding_size, forget_bias, weights):
        self._forget_bias = forget_bias
        self._w = weights

    def __call__(self, state, inputs):
        c, h = state
        max_seq_len = tf.shape(inputs)[0]
        print('bs', batch_size)
        print('inputs', inputs)
        print('h', h)
        print('tf ones', tf.ones([batch_size, 1]))
        # inputs: [batch_size, embedding_size]
        # h: [batch_size, embedding_size]
        # ones: [batch_size, 1]
        linear_in = tf.concat([inputs, h, tf.ones([max_seq_len, 1])], axis=1)
        linear_out = tf.matmul(linear_in, self._w)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=linear_out, num_or_size_splits=4, axis=1)

        new_c = (c * tf.sigmoid(f + self._forget_bias) +
                 tf.sigmoid(i) * tf.tanh(j))
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return new_c, new_h


def bayesian_rnn(cell, inputs, y_i):
    batch_size = tf.shape(inputs)[1]
    print('encore batch size', batch_size)
    initializer_c_h = (tf.zeros([batch_size, embedding_size]), tf.zeros([batch_size, embedding_size]))
    c_list, h_list = tf.scan(cell, inputs, initializer=initializer_c_h)
    # return h_list

    # If we're only interested in the last state
    # indices = tf.stack([seq_len, tf.range(batch_size)], axis=1)
    # relevant_outputs = tf.gather_nd(h_list, indices)
    # logits = tf.squeeze(tf.layers.dense(relevant_outputs, 1), -1)

    # logits = tf.layers.dense(h_list, (nb_items * nb_classes))  # Not possible so we have to implement our own tensor product
    Wo = tf.get_variable('weight', shape=(embedding_size, nb_items, nb_classes))
    # Well bias is missing, I'm lazy now, I just want the type check to shut up

    logits = tf.tensordot(h_list, Wo, axes=1)  # shape = [max_seq_len, nb_batches, nb_items, nb_classes]
    slicer = tf.one_hot(y_i, depth=nb_items)  # shape = [max_seq_len, nb_batches, nb_items (one-hot)]
    relevant_logits = tf.einsum('ijkl,ijk->ijl', logits, slicer)  # shape = [max_seq_len, nb_batches, nb_classes]
    return relevant_logits

# Data
embedding_size = 128
# None = 2
nb_classes = 10

# Training
batch_size = 20
nb_epochs = 10
nb_mc_samples = 2

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

with open('ml0.pickle', 'rb') as f:
    data = pickle.load(f)
    nb_items = pickle.load(f)

nb_users = len(data)
seq_lengths = np.array(list(map(len, data))) - 1
max_seq_len = max(seq_lengths)

x_all = np.zeros((max_seq_len, len(data), embedding_size))
y_i_all = np.zeros((max_seq_len, len(data)))
y_v_all = np.zeros((max_seq_len, len(data)))
for i_sample, sample in enumerate(data):
    for pos, (i, v) in enumerate(sample[:-1]):
        if (i, v) not in items:
            items[i, v] = np.random.random(embedding_size)
        x_all[pos, i_sample] = items[i, v]
    indices, values = zip(*sample[1:])
    # print(indices, values, seq_lengths[i_sample], y_i_train.shape, y_v_train.shape)
    y_i_all[:seq_lengths[i_sample], i_sample] = indices
    y_v_all[:seq_lengths[i_sample], i_sample] = values
y_i_all = np.array(y_i_all)
y_v_all = np.array(y_v_all)

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
x = tf.placeholder(tf.float32, shape=[None, None, embedding_size], name='x')
y_i = tf.placeholder(tf.int32, shape=[None, None], name='y_i')
y_v = tf.placeholder(tf.int32, shape=[None, None], name='y_v')
seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')

def make_decoder(sampled_weights):
    '''
    Decoder: p(x|z) = p(y_v|w)
    '''
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        cell = BayesianLSTMCell(128, forget_bias=0., weights=sampled_weights)
        # shape was [max_seq_len, nb_batches, nb_classes]
        relevant_logits = bayesian_rnn(cell, x, y_i)
        print('relevant', relevant_logits)
        # item_features = tf.get_variable("item_features", shape=[nb_items, embedding_size, nb_classes],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        # relevant_items = tf.nn.embedding_lookup(item_features, y_i, name="feat_items")
        # print('hl', h_list)
        # print('ri', relevant_items)
        # logits = tf.tensordot(h_list, relevant_items, axes=[[2], [2]])  # That's not even the good shape but anyway
        y = tfd.Categorical(logits=relevant_logits)  # shape of its local_log_prob = [max_seq_len, nb_batches]
                                           # because we already observe the true variable (y_v is in observed)
        print('y', y)
        return y, relevant_logits

def make_encoder():
    w_mean = tf.get_variable(
        'w_mean', shape=[embedding_size * 2 + 1, 4 * embedding_size],
        initializer=tf.constant_initializer(0.))
    w_logstd = tf.get_variable(
        'w_logstd', shape=[embedding_size * 2 + 1, 4 * embedding_size],
        initializer=tf.constant_initializer(-3.))
    return tfd2.Normal(loc=w_mean, scale=tf.nn.softplus(w_logstd))



def compute_elbo():
    q_net = make_encoder()
    prior = make_prior()
    ws = q_net.sample(nb_mc_samples)

    score = []
    for i_sample in range(nb_mc_samples):
        p_net, relevant_logits = make_decoder(ws[i_sample])
        # print('all', model._stochastic_tensors)  # w and y_v
        # log_pz, log_px_z = model.local_log_prob(['w', 'y_v'])  # Error
        # print('poids', cell._w)
        print('values', y_v)
        log_pz = tf.reduce_sum(prior.log_prob(ws[i_sample]))
        # log_pz = 0
        log_px_z = fetch_relevant_mean(p_net.log_prob(y_v), seq_len)
        print('log prior', log_pz)
        print('log likelihood', log_px_z)
        # log_px_z = model.local_log_prob('y_v')
        log_qz_x = q_net.log_prob(ws[i_sample])
        score.append(log_pz + log_px_z - log_qz_x)

    return tf.reduce_mean(score), relevant_logits  # Error
    # return log_px_z

elbo, logits = compute_elbo()
print('logits', logits)
cost = -elbo

# If I had all relevant_logits (shape: [max_seq_len, nb_batches, nb_classes]) I would do this:
# labels = tf.one_hot(y_v, depth=nb_classes)
# cost = tf.nn.softmax_cross_entropy_with_logits_v2(
#         labels=labels,
#         logits=relevant_logits)

optimizer = tf.train.AdamOptimizer(0.1)
infer_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, nb_epochs + 1):
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

        lb_test, test_logits = sess.run([elbo, logits],
                         feed_dict={x: x_test, y_i: y_i_test, y_v: y_v_test, seq_len: seq_lengths_test})
        acc, test_rmse, test_macc = compute_metrics(y_v_test, test_logits, seq_lengths_test)
        # print(acc.eval()[:10])
        print('Epoch {}: Lower bound = {} Test error = {} RMSE={:f} ACC={:f}'.format(
              epoch, np.sum(lbs), lb_test, test_rmse.eval(), test_macc.eval()))

    lb, test_logits = sess.run([elbo, logits],
                                  feed_dict={x: x_test, y_i: y_i_test, y_v: y_v_test, seq_len: seq_lengths_test})
    acc, test_rmse, test_macc = compute_metrics(y_v_test, test_logits, seq_lengths_test)
    test_proba = tf.nn.softmax(test_logits)
    print('test proba', test_proba)
    print('seq len', seq_lengths_test)
    test_pred = tf.argmax(test_proba, 2)
    print('Test error', lb)
    print('snif', test_rmse.eval(), test_macc.eval())
    for i in range(x_test.shape[1]):
        print(np.round(test_pred[:seq_lengths_test[i], i].eval(), 3))
        print(y_v_test[:seq_lengths_test[i], i])

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/train', sess.graph)
    test_writer = tf.summary.FileWriter('/tmp/test')
