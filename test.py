import tensorflow as tf
import tensorflow.distributions as tfd
import tensorflow.contrib.distributions as wtfd

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
