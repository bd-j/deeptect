#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
#import matplotlib.pyplot as pl

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model(nside=64, training=True):
    """Build model functionally
    """
    inputs = keras.Input(shape=(nside, nside, 1))
    conv_kwargs = dict(strides=1, activation="relu",
                       padding="same")

    x = layers.Conv2D(16, 8, **conv_kwargs)(inputs)
    x = layers.Dropout(0.25)(x, training=training)
    x = layers.Conv2D(32, 4, **conv_kwargs)(x)
    x = layers.Conv2D(64, 2, **conv_kwargs)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    deeptect = Model(inputs=inputs, outputs=outputs)

    return deeptect


def train_model(model, train_X, train_Y, test_X, test_Y,
                epochs=50, save_loc="./weights.h5"):

    es = EarlyStopping(monitor='val_loss', patience=5)
    weight_save_callback = ModelCheckpoint(save_loc, monitor='val_loss', verbose=0,
                                           save_best_only=True, mode='auto')
    print(weight_save_callback)

    #adadelta = keras.optimizers.adadelta(lr=1.0, decay=0.0, rho=0.99)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy')

    start = time.time()
    out = model.fit(train_X, train_Y,
                    epochs=epochs, batch_size=128, shuffle=True,
                    validation_data=(test_X, test_Y),
                    callbacks=[es, weight_save_callback])
    end = time.time()
    print(end-start)

    loss = out.history['loss']
    val_loss = out.history['val_loss']
    epochs = range(len(loss))

    return out


class DeepTect(keras.models.Model):

    def __init__(self, config):
        super(DeepTect, self).__init__()

        self.input_shape = config.input_shape
        self.build(config)

    def build(self, config):
        nfilt = config.convolutional_filters
        sizes = config.convolutional_sizes

        convs = [layers.Conv2D(nf, size, **conv_kwargs)
                 for nf, size in zip(nfilt, sizes)]

    def call(self, input, training=True):
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    pass