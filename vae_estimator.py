from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import tensorflow as tf
import pandas as pd


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed



class VAEGenerator:
    def __init__(self, layer_Ns = [12, 8, 4]):
        self._decoder = None
        self._encoder = None
        
        self._vae_fitted = False
        self.layer_Ns = layer_Ns
        self._latent_dim = layer_Ns[-1]
        self._input_dim = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self._X_train_scaled = None
        self._X_test_scaled = None
        self._scaler = MinMaxScaler()

    def _build_encoder(self, input_dim, latent_dim):
        inputs = layers.Input(shape=input_dim)
        x = layers.Dense(12, activation="relu")(inputs)
        x = layers.Dense(8, activation="relu")(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder


    def _build_decoder(self, latent_dim, input_dim):
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(8, activation="relu")(latent_inputs)
        x = layers.Dense(12, activation="relu")(x)
        outputs = layers.Dense(np.prod(input_dim), activation="sigmoid")(x)
        decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
        return decoder    

    def load_data(self, X: pd.DataFrame):

        self.X = X
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self._X_train_scaled = self._scaler.fit_transform(self.X_train)
        self._X_test_scaled = self._scaler.transform(self.X_test)

        self._input_dim = self.X.shape[1:]

        self._encoder = self._build_encoder(self._input_dim, self._latent_dim)
        self._decoder = self._build_decoder(self._latent_dim, self._input_dim)
        

        self.vae = VAE(self._encoder, self._decoder)
        self.vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())


    def fit(self, **kwargs):
        epochs = kwargs.get("epochs", 120)
        if self._input_dim is None:
            raise ValueError("NO DATA HAVE BEEN LOADED!")
        else:
            self.vae.fit(self._X_train_scaled, self._X_train_scaled, epochs=epochs, batch_size=254, validation_data=(self._X_test_scaled, self._X_test_scaled))
            self._vae_fitted = True

    def generate(self, n = 10000):
        if self._vae_fitted:
            z_sample = np.random.normal(size=(n, self._latent_dim))
            x_decoded = self._decoder.predict(z_sample)
            arr  = self._scaler.inverse_transform(x_decoded)
            return pd.DataFrame(arr, columns = self.X.columns)
        else:
            raise NotFittedError("MODEL NOT FITTED YET!")