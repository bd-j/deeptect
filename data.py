#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import argparse
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits


def split(full, nside, everypixel=False):
    """Split an image into square tiles.
    These can either be non-overlapping or centered on every input pixel.

    TODO: deal with images with dimensions that are non-integer multiples of the stamp size
    """
    from numpy.lib.stride_tricks import as_strided
    bs = full.itemsize
    sx, sy = full.shape
    assert np.mod(sx, nside) == 0
    assert np.mod(sy, nside) == 0
    nx, ny = sx // nside, sy // nside
    if everypixel:
        s = as_strided(full, shape=(nside, nside, sx * sy),
                       strides=(bs*sy, bs, bs))
    else:
        s = as_strided(full, shape=(nside, nside, nx * ny),
                       strides=(bs*sy, bs, bs*nside))
    return s


def prep_data_x(files, nside=64):
    channel = []
    for f in files:
        signal = fits.getdata(f, 1)
        noise = fits.getdata(f, 3)
        snr = signal / noise

        stamps = split(snr, nside)
        channel.append(stamps.T)
    return np.array(channel).astype(np.float32)


def prep_data_y(files, nside=64):
    channel = []
    for f in files:
        truth = fits.getdata(f, 4)
        stamps = split(truth, nside)
        channel.append(stamps.T)
    return np.array(channel).astype(np.float32)


def training_data(files, nside=64):

    X = prep_data_x(files, nside=nside)
    Y = prep_data_y(files, nside=nside)

    train_X = X.reshape((-1, nside, nside))
    train_Y = Y.reshape((-1, nside, nside))

    return train_X, train_Y


if __name__ == "__main__":

    config = argparse.Namespace()
    config.data_dir = "./data/training_data_20210305"
    config.n_train = 50
    config.n_test = 10

    search = os.path.join(config.data_dir, "training_data_*3*fits")

    files = glob.glob(search)
    train_X, train_Y = training_data(files[:config.n_train])
    test_X, test_Y = training_data(files[config.n_train:(config.n_train + config.n_test)])

