#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as pl

from astropy.io import fits

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


def decimate(full, nside):
    from numpy.lib.stride_tricks import as_strided
    bs = full.itemsize
    sx, sy = full.shape
    assert np.mod(sx, nside) == 0
    assert np.mod(sy, nside) == 0
    nx, ny = sx // nside, sy // nside
    s = as_strided(full, shape=(nside, nside, nx * ny),
                   strides=(bs*sy, bs, bs*nside))
    return s


def prep_data(files, nside=64):
    channel = []
    for f in files:
        signal = fits.getdata(f, 0)
        noise = fits.getdata(f, 1)
        snr = signal / noise

        stamps = decimate(snr, nside)
        channel.append(stamps.T)
    return np.array(channel)



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