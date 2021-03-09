#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import argparse
import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits
from skimage.measure import block_reduce


def split(full, nside, everypixel=False, block=1):
    """Split an image into square tiles.
    These can either be non-overlapping or centered on every input pixel.

    TODO: deal with images with dimensions that are non-integer multiples of the stamp size
    """
    from numpy.lib.stride_tricks import as_strided
    bs = full.itemsize
    sx, sy = full.shape
    if nside == sx:
        return full[:, :, None]
    assert np.mod(sx, nside) == 0
    assert np.mod(sy, nside) == 0
    nx, ny = sx // nside, sy // nside
    #if everypixel:
    #    s = as_strided(full, shape=(nside, nside, sx * sy),
    #                   strides=(bs*sy, bs, bs))
    #else:
    #    s = as_strided(full, shape=(nside, nside, nx * ny),
    #                   strides=(bs*sy, bs, bs*nside))

    s = []
    for i in range(nx):
        for j in range(ny):
            sub = full[i * nside:(i+1)*(nside), j*nside:(j+1)*(nside)]
            if block > 1:
                sub = block_reduce(sub, (block, block), func=np.sum)
            s.append(sub)

    return np.array(s).T


def prep_data_x(files, n_side=64, block=1):
    channel = []
    for f in files:
        signal = fits.getdata(f, 1)
        noise = fits.getdata(f, 3)
        snr = signal / noise

        stamps = split(snr, n_side, block=block)
        channel.append(stamps.T)
    return np.array(channel).astype(np.float32)


def prep_data_y(files, n_side=64, snr_limit=10, block=1):
    channel = []
    for f in files:
        truth = fits.getdata(f, 4)
        cat = fits.getdata(f, 5)
        bad = cat["snr"] < snr_limit
        xg, yg = cat[bad]["x"], cat[bad]["y"]
        for x, y in zip(xg, yg):
            truth[int(y), int(x)] = 0
        stamps = split(truth, n_side, block=block)
        channel.append(stamps.T)
    return np.array(channel).astype(np.float32)


def training_data(files, n_side=64, snr_limit=10, block=1):

    X = prep_data_x(files, n_side=n_side, block=block)
    Y = prep_data_y(files, n_side=n_side, snr_limit=snr_limit, block=block)

    train_X = X.reshape((-1, n_side // block, n_side // block))
    train_Y = Y.reshape((-1, n_side // block, n_side // block))

    return train_X[:, :, :, None], train_Y[:, :, :, None]


if __name__ == "__main__":

    config = argparse.Namespace()
    config.data_dir = "./data/training_data_20210305"
    config.n_train = 50
    config.n_test = 10

    search = os.path.join(config.data_dir, "training_data_*3*fits")

    files = glob.glob(search)
    train_X, train_Y = training_data(files[:config.n_train])
    test_X, test_Y = training_data(files[config.n_train:(config.n_train + config.n_test)])

