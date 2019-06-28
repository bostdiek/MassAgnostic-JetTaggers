# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
project_dir = Path(__file__).resolve().parents[2]
# print(project_dir.resolve())
interim_datadir = str(project_dir.resolve()) + '/data/interim/'
pred_datadir = str(project_dir.resolve()) + '/data/modelpredictions/'


def load_data_bdt(prong, set='train', scale='normal'):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
        set: should either be "train" or "test"
    Outputs:
        df: pandas dataframe with the scaled data and mass (to flatten)
        y: np.array of training labels
        features: mass + names of columns to train on
    '''
    data_cols = ['(pruned)m', 'scaled_tau_(1)^(1/2)',  # 'pT',
                 'scaled_tau_(1)^(1)', 'scaled_tau_(1)^(2)',
                 'scaled_tau_(2)^(1/2)', 'scaled_tau_(2)^(1)', 'scaled_tau_(2)^(2)',
                 'scaled_tau_(3)^(1/2)', 'scaled_tau_(3)^(1)', 'scaled_tau_(3)^(2)',
                 'scaled_tau_(4)^(1)', 'scaled_tau_(4)^(2)'
                 ]

    #  Training data
    if scale == 'normal':
        x_name = set + '_scaled_X_{0}p.npy'.format(prong)
    elif scale == 'pca':
        x_name = set + '_X_PCA_{0}p.npy'.format(prong)
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


def load_data_PCA_nn(prong):
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
    train_x_name = 'train_X_PCA_{0}p.npy'.format(prong)
    X_trainscaled = np.load(interim_datadir + train_x_name)
    y_train_name = 'train_Y_{0}p.npy'.format(prong)
    y_train = np.load(interim_datadir + y_train_name)

    # Validation data
    val_x_name = 'val_X_PCA_{0}p.npy'.format(prong)
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


def load_test_data_PCA_nn(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        X_testscaled: numpy array of test data scaled and centered
        y_test: test labels
        mass: numpy array for the jet masses
    '''
    test_x_name = 'test_X_PCA_{0}p.npy'.format(prong)
    X_testscaled = np.load(interim_datadir + test_x_name)

    y_test_name = 'test_Y_{0}p.npy'.format(prong)
    y_test = np.load(interim_datadir + y_test_name)

    m_name = 'test_jetmass_{0}p.npy'.format(prong)
    mass = np.load(interim_datadir + m_name)

    return X_testscaled, y_test, mass


def load_test_data_unscaled(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        X_test: numpy array of test data with no scaling
        y_test: test labels
        mass: numpy array for the jet masses
        pt: numpy array for the jet transverse momentum
    '''
    test_x_name = 'test_X_{0}p.npy'.format(prong)
    X_test = np.load(interim_datadir + test_x_name)

    y_test_name = 'test_Y_{0}p.npy'.format(prong)
    y_test = np.load(interim_datadir + y_test_name)

    m_name = 'test_jetmass_{0}p.npy'.format(prong)
    mass = np.load(interim_datadir + m_name)

    pt_name = 'test_jetpT_{0}p.npy'.format(prong)
    pt = np.load(interim_datadir + pt_name)

    return X_test, y_test, mass, pt


def write_roc_pickle(file_name, roc_info):
    '''
    Saves the ROC curve information to a pickle file
    Inputs:
        file_name: give the model name and prong as a string
        roc_info: tuple correspond to (fpr, tpr, thresholds, auc)
    Outputs:
        Dictionary that was saved
    '''
    fpr, tpr, thresholds, auc = roc_info
    full_name = pred_datadir + file_name + '_roc.p'

    # make the dictionary to be written
    data = {'x_data': tpr[fpr > 0],
            'y_data': 1.0 / fpr[fpr > 0],
            'cut_values': thresholds[fpr > 0],
            'auc': auc
            }

    with open(full_name, 'wb') as f:
        pickle.dump(data, f)
    return data


def make_histos(model_name, jet_mass, predicted_probabilities, y_true, roc_info):
    '''
    Writes the files a pickle file for the jet mass of signal and background
    events at different fixed signal efficiencies.

    Inputs:
        model_name: model name of the file to be written
        predicted_probabilities: preictions of the model
        y_tue: true labels that are trying to be predicted
        roc_info: tupple of (fpr, tpr, thresholds, auc)
    Outputs:
        histo_dictionary: dictionary with efficiencies as keys.
            2D tupple of
                (np.array(signal masses passing),
                 np.array(background masses passing)
                )
            for the value
    '''
    full_name = pred_datadir + model_name + '_histos.p'

    fpr, tpr, thresholds, auc = roc_info

    sig_mass = jet_mass[np.ravel(y_true == 1)]
    bkg_mass = jet_mass[np.ravel(y_true == 0)]

    sig_pred = predicted_probabilities[np.ravel(y_true == 1)]
    bkg_pred = predicted_probabilities[np.ravel(y_true == 0)]

    histo_dictionary = OrderedDict()
    efficiencies = np.linspace(0.05, 1, 96)

    for eff in efficiencies:
        eff = np.round(eff, decimals=2)
        # get the index of the tpr closest to the signal efficiency requested
        i = np.argmin(np.abs(tpr - eff))
        fpr_i, tpr_i, thr_i = fpr[i], tpr[i], thresholds[i]
        if eff == 1:
            thr_i = -1
        sig_mass_pass = sig_mass[sig_pred > thr_i]
        bkg_mass_pass = bkg_mass[bkg_pred > thr_i]
        histo_dictionary[eff] = (sig_mass_pass, bkg_mass_pass)

    # write the file to disk
    with open(full_name, 'wb') as f:
        pickle.dump(histo_dictionary, f)

    return histo_dictionary
