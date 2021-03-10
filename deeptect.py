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


def rectify(Xes, med=None, std=None, mean=None, ftype="linear"):

    oX = []
    for X in Xes:
        if ftype == "log":
            X = np.clip(X, 1e-3, np.inf)
            X = np.log(X)
        if med is None:
            med = np.median(X)
        if std is None:
            std = np.std(X)
        if mean is None:
            mean = np.mean(X)
        if ftype == "log":
            X -= med
            X /= std
        else:
            X /= mean
        assert np.all(np.isfinite(X))
        oX.append(X)

    print(f"median is {med}, std is {std}")

    return oX



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="./data/training_data_20210309")
    parser.add_argument("--n_side", type=int, default=128)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--frac_test", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--snr_limit", type=int, default=10)
    parser.add_argument("--renorm", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    config = parser.parse_args()
    config.n_test = int(config.n_train * config.frac_test)
    ftype = "log"

    search = os.path.join(config.data_dir, "training_data_*fits")

    files = glob.glob(search)
    train_ims = files[:config.n_train]
    test_ims = files[config.n_train:(config.n_train + config.n_test)]
    start, stop = (config.n_train + config.n_test), (config.n_train + config.n_test + 10)
    val_ims = files[start:stop]

    train_X, train_Y = training_data(train_ims, n_side=config.n_side,
                                     snr_limit=config.snr_limit, block=config.block)
    test_X, test_Y = training_data(test_ims, n_side=config.n_side,
                                   snr_limit=config.snr_limit, block=config.block)
    val_X, val_Y = training_data(val_ims, n_side=config.n_side,
                                 snr_limit=config.snr_limit, block=config.block)

    deeptect = build_model(config.n_side // config.block)

    if config.renorm:
        inx = train_X.max()
        train_X, test_X, val_X = rectify([train_X, test_X, val_X], ftype=ftype)
        print(inx, train_X.max())
        assert train_X.max() < inx

    if config.no_training:
        sys.exit()

    history = train_model(deeptect, train_X, train_Y, test_X, test_Y,
                          epochs=config.epochs)

    deeptect.save(f"{ftype}_flux.h5", save_format="h5")
    pred_Y = deeptect.predict_on_batch(val_X)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(f"{ftype}_flux_validation.pdf") as pdf:
        for i in range(100, 500, 25):
            fig, ax = show_pred(i, val_X[..., 0], val_Y[..., 0], pred_Y)
            fig.set_size_inches(9.0, 6.5)
            pdf.savefig(fig)
            pl.close(fig)
