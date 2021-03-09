#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    config = argparse.Namespace()
    config.data_dir = "./data/training_data_20210305"
    config.n_side = 64
    config.n_train = 50
    config.n_test = 10

    search = os.path.join(config.data_dir, "training_data_*3*fits")

    files = glob.glob(search)
    train_ims = files[:config.n_train]
    test_ims = files[config.n_train:(config.n_train + config.n_test)]
    train_X, train_Y = training_data(train_ims, n_side=config.n_side)
    test_X, test_Y = training_data(test_ims, n_side=config.n_side)

    deeptect = build_model(config.n_side)

    history = train_model(deeptect, train_X, train_Y, test_X, test_Y)

    start, stop = (config.n_train + config.n_test), (config.n_train + config.n_test + 5)
    val_ims = files[start:stop]
    val_X, val_Y = training_data(val_ims, n_side=config.n_side)
    pred_Y = deeptect.predict_on_batch(val_ims)