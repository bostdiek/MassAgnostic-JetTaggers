# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle

interim_datadir = 'data/interim/'


def load_data_bdt(prong, set='train'):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
        set: should either be "train" or "test"
    Outputs:
        df: pandas dataframe with the scaled data and mass (to flatten)
        y: np.array of training labels
        features: mass + names of columns to train on
    '''
    data_cols = ['(pruned)m', 'pT', 'scaled_tau_(1)^(1/2)',
                 'scaled_tau_(1)^(1)', 'scaled_tau_(1)^(2)',
                 'scaled_tau_(2)^(1/2)', 'scaled_tau_(2)^(1)', 'scaled_tau_(2)^(2)',
                 'scaled_tau_(3)^(1/2)', 'scaled_tau_(3)^(1)', 'scaled_tau_(3)^(2)',
                 'scaled_tau_(4)^(1)', 'scaled_tau_(4)^(2)'
                 ]

    #  Training data
    x_name = set + '_scaled_X_{0}p.npy'.format(prong)
    X_scaled = np.load(interim_datadir + x_name)

    y_name = set + '_Y_{0}p.npy'.format(prong)
    y = np.load(interim_datadir + y_name)

    mass_name = set + '_jetmass_{0}p.npy'.format(prong)
    mass = np.load(interim_datadir + mass_name).reshape(-1, 1)

    #  Combine mass and features
    X = np.hstack((mass, X_scaled))
    df = pd.DataFrame(X,
                      columns=data_cols
                      )

    return df, y, data_cols


def load_data_nn(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        X_trainscaled: numpy array of training data scaled and centered
        y_train: training labels
        X_valscaled: numpy array of validation data scaled and centered
        y_val: validation labels
        val_weights: class weights applied to each element of the validation set
        class_weights: dictionary to set balanced classes
    '''
    #  Training data
    train_x_name = 'train_scaled_X_{0}p.npy'.format(prong)
    X_trainscaled = np.load(interim_datadir + train_x_name)
    y_train_name = 'train_Y_{0}p.npy'.format(prong)
    y_train = np.load(interim_datadir + y_train_name)

    # Validation data
    val_x_name = 'val_scaled_X_{0}p.npy'.format(prong)
    X_valscaled = np.load(interim_datadir + val_x_name)
    y_val_name = 'val_Y_{0}p.npy'.format(prong)
    y_val = np.load(interim_datadir + y_val_name)
    val_weight_name = 'val_class_weights_indiv_{0}p.npy'.format(prong)
    val_weights = np.load(interim_datadir + val_weight_name)

    # classes info
    with open(interim_datadir + 'class_weights_{0}p.p'.format(prong), 'rb') as cf:
        class_weights = pickle.load(cf)

    return (X_trainscaled, y_train,
            X_valscaled, y_val, val_weights,
            class_weights
            )


def load_test_data_nn(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        X_testscaled: numpy array of test data scaled and centered
        y_test: test labels
        mass: numpy array for the jet masses
    '''
    test_x_name = 'test_scaled_X_{0}p.npy'.format(prong)
    X_testscaled = np.load(interim_datadir + test_x_name)

    y_test_name = 'test_Y_{0}p.npy'.format(prong)
    y_test = np.load(interim_datadir + y_test_name)

    m_name = 'test_jetmass_{0}p.npy'.format(prong)
    mass = np.load(interim_datadir + m_name)

    return X_testscaled, y_test, mass
