from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import argparse
import matplotlib.pyplot as plt

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

## VAE 
class VAE(object):
    # initialize class value for later processing purpose
    def __init__(self, original_dim, intermediate_dim, latent_dim, batch_size=128, epochs=50):
        self.input_shape = (original_dim, )
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder = None
        self.decoder = None
        self.VAE_model = None
    
    def instantiate_encoder(self):
        # build encoder model
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
    def instantiate_decoder(self):
        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        
    # instantiate VAE model
    def instantiate_VAE(self, loss='mse'):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        outputs = self.decoder(self.encoder(inputs)[2])
        
        self.VAE_model = Model(inputs, outputs, name='vae_mlp')
        
        if loss == 'mse':
            reconstruction_loss = mse(inputs, outputs)
        elif loss == 'cross_entropy':
             reconstruction_loss = binary_crossentropy(inputs, outputs)
        else:
            raise ValueError('Loss selected not found...')
            
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.VAE_model.add_loss(vae_loss)
        self.VAE_model.compile(optimizer='adam')
    
    def fit(self, x_train, x_test):
        # train the autoencoder
        self.VAE.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    
    def get_encoder(self, dsp_type):
        if dsp_type == "spec":
            self.encoder.summary()
        elif dsp_type == "image":
            plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        
    def get_decoder(self, dsp_type):
        if dsp_type == "spec":
            self.decoder.summary()
        elif dsp_type == "image":
            plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
            
    def get_VAE(self, dsp_type):
        if dsp_type == "spec":
            self.VAE_model.summary()
        elif dsp_type == "image":
            plot_model(self.VAE_model, to_file='vae.png', show_shapes=True)