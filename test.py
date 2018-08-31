import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow.contrib.distributions as wtfd
from scipy.sparse import coo_matrix, load_npz, find
import numpy as np


tf.enable_eager_execution()

embedding_size = 5

def make_prior():
    return tfd.Normal(loc=[0.] * embedding_size, scale=[1.] * embedding_size)

def make_other_prior():
    return wtfd.MultivariateNormalDiag(loc=[0.] * embedding_size, scale_diag=[1.] * embedding_size)

prior = make_prior()
prior2 = make_other_prior()
draw = [1., 2., 3., 4., 5.]

test = prior.log_prob(draw)
print(test)
print(tf.reduce_sum(test))
print(test.numpy().sum())
print(prior2.log_prob(draw))

# X_fm_batch = tf.sparse_placeholder(tf.int32, shape=[None, nb_users + nb_items], name='sparse_batch')
# outcomes = tf.placeholder(tf.float32, shape=[None], name='outcomes')

X = load_npz('data/mangaki/X_fm.npz')
rows, cols, data = find(X)
wow = tf.SparseTensorValue(np.column_stack((rows, cols)), X.shape, data)
#print(tf.constant(wow))
print(wow)
print(wow[0])

# print(prior.cdf(1.7))
# for _ in range(3):
#     print(prior.sample([2]))
