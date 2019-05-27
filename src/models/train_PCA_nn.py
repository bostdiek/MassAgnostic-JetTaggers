# -*- coding: utf-8 -*-
'''
Author: Andrea Mitridate; Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/interm directory
Trains and saves the basic neural network
'''
import click

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input
from keras.models import Model

import logging
import numpy as np
from pathlib import Path
import pickle

from HelperFunctions import load_data_PCA_nn


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def train_PCA_NN(prong):
    data = load_data_PCA_nn(prong)
    X_trainscaled = data[0]
    y_train = data[1]
    X_valscaled = data[2]
    y_val = data[3]
    val_weights = data[4]
    class_weights = data[5]

    #  tr_class_weights data
    tr_class_weights = np.ones_like(y_train, dtype='float')
    tr_class_weights[y_train == 0] = class_weights[0]
    tr_class_weights[y_train == 1] = class_weights[1]
    tr_class_weights = tr_class_weights.flatten()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                                  patience=5, min_lr=1.0e-6)
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    inputs = Input(shape=(X_trainscaled.shape[1], ))
    Classifier = Dense(50, activation='relu')(inputs)
    Classifier = Dense(50, activation='relu')(Classifier)
    Classifier = Dense(50, activation='relu')(Classifier)
    Classifier = Dense(1, activation='sigmoid')(Classifier)
    ClassifierModel = Model(inputs=inputs, outputs=Classifier)

    ClassifierModel.compile(optimizer='adam', loss='binary_crossentropy')
    ClassifierModel.summary()
    history = ClassifierModel.fit(X_trainscaled,
                                  y_train,
                                  validation_data=[X_valscaled, y_val, val_weights],
                                  epochs=100,
                                  callbacks=[reduce_lr, es],
                                  sample_weight=tr_class_weights
                                  )
    ClassifierModel.save('models/pca_nn_{0}p.h5'.format(prong))

    with open('models/histories/pca_nn_hist_{0}p.p'.format(prong), 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    train_PCA_NN()
