'''
Illustrating example about how modern coding of TF2.9 can be a pain in the ass.
This was an attempt at VAE.
Thanks to Xizewen Han's blog post on Medium.
Author: Jill-Jênn Vie, 2022
'''
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import numpy as np
tfd = tfp.distributions


N_EPOCHS = 100
Z_DIM = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.01


def _preprocess(sample):  # This is also ridiculous
    '''
    Note that _preprocess() above returns image, image rather than just image
    because Keras is set up for discriminative models with an (example, label)
    input format, i.e. (p\theta(y|x)). Since the goal of the VAE is to recover
    the input x from x itself (i.e. ), the data pair is (example, example).
    From: https://tensorflow.org/probability/examples/Probabilistic_Layers_VAE
    '''
    return sample, sample


x_train = np.load('fraction.npy')
print('Fraction data loaded', x_train.shape)
_, x_dim = x_train.shape
train_dataset = tf.data.Dataset.from_tensor_slices(
    x_train).map(_preprocess).batch(BATCH_SIZE).shuffle(1000)


class VAE:
    def __init__(self):
        prior = tfd.MultivariateNormalDiag(loc=tf.zeros(Z_DIM),
                                           scale_diag=tf.ones(Z_DIM))
        self.encoder = tf.keras.Sequential([  # q_net
            layers.Dense(5),
            # TensorFlow I hate you so much, what follows makes no sense to me
            layers.Dense(tfp.layers.IndependentNormal.params_size(Z_DIM),
                         activation=None, name='z_params'),
            tfp.layers.IndependentNormal(
                Z_DIM,
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior),
                convert_to_tensor_fn=tfd.Distribution.sample, name='z_layer')
        ], name='encoder')
        self.decoder = tf.keras.Sequential([  # p_net
            layers.Dense(5),
            layers.Dense(x_dim),
            tfp.layers.IndependentBernoulli(x_dim)
        ], name='decoder')

    def build(self):
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        negloglik = lambda x, rv_x: -rv_x.log_prob(x)  # Couldn't find doc
        x_input = tf.keras.Input(shape=(x_dim,))
        z = self.encoder(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=self.decoder(z))
        model.compile(loss=negloglik, optimizer=optimizer)
        return model


model = VAE().build()
model.fit(train_dataset, epochs=N_EPOCHS)
