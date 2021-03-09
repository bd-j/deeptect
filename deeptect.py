#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os, glob
import argparse
import numpy as np
import matplotlib.pyplot as pl

from model import build_model, train_model
from data import training_data


def show_pred(ind, val_X, val_Y, pred_Y):
    import matplotlib.pyplot as pl
    pl.ion()
    fig, axes = pl.subplots(2, 2)
    ax = axes.flat[0]
    ax.imshow(val_X[ind].T)
    ax = axes.flat[1]
    ax.imshow(val_Y[ind].T)
    ax = axes.flat[3]
    ax.imshow(pred_Y[ind, :, :, 0].T)
    return fig, axes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./data/training_data_20210305")
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--ntrain", type=int, default=100)
    parser.add_argument("--n_test", type=int, defailt=20)
    parser.add_argument("--epochs", type=int, defailt=20)
    parser.add_argument("--snr_limit", type=int, default=10)
    parser.add_argument("--renorm", action="store_true")
    config = parser.parse_args()

    search = os.path.join(config.data_dir, "training_data_*3*fits")

    files = glob.glob(search)
    train_ims = files[:config.n_train]
    test_ims = files[config.n_train:(config.n_train + config.n_test)]
    train_X, train_Y = training_data(train_ims, n_side=config.n_side,
                                     snr_limit=config.snr_limit)
    test_X, test_Y = training_data(test_ims, n_side=config.n_side,
                                   snr_limit=config.snr_limit)

    if config.renorm:
        scale = train_X.max()
        print(f"scale = {scale}")
    else:
        scale = 1.0

    train_X /= scale
    test_X /= scale

    deeptect = build_model(config.n_side)

    #sys.exit()

    history = train_model(deeptect, train_X, train_Y, test_X, test_Y,
                          epochs=config.epochs)

    start, stop = (config.n_train + config.n_test), (config.n_train + config.n_test + 10)
    val_ims = files[start:stop]
    val_X, val_Y = training_data(val_ims, n_side=config.n_side,
                                 snr_limit=config.snr_limit)
    val_X /= scale
    pred_Y = deeptect.predict_on_batch(val_X)

    from tensorflow.keras.losses import binary_crossentropy
