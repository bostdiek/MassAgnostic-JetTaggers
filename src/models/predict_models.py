# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/interm directory
Loads models previously saved
Makes predictions and saves them to the data/modelpredictions directory
'''
import click
from collections import OrderedDict
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
import pickle
from pathlib import Path
import logging

from HelperFunctions import load_data_bdt, load_test_data_nn
from HelperFunctions import write_roc_pickle, make_histos


def predict_uboost(prong):
    '''
    Computes the predictions for the uBoost method.
    Inputs:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    testdf, y_test, data_cols = load_data_bdt(prong, set='test')
    jet_mass = testdf[data_cols[0]].values
    name = str(project_dir) + '/models/uBoost_{0}p.p'.format(prong)

    # load the model and make predictions
    uboost = joblib.load(name)
    sig_prob = uboost.predict_proba(testdf)[:, 1]

    # compute the test statistics
    fpr_u, tpr_u, thresholds_u = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_u = auc(fpr_u, tpr_u)
    roc_info = fpr_u, tpr_u, thresholds_u, auc_u
    # save the info
    model_name = 'uBoost_{0}p'.format(prong)
    write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos


def predict_gbc(prong):
    '''
    Computes the predictions for the gradient boosted decision tree.
    Inputs:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    testdf, y_test, data_cols = load_data_bdt(prong, set='test')
    jet_mass = testdf[data_cols[0]].values
    name = str(project_dir) + '/models/GBC_{0}p.p'.format(prong)

    # load the model and make predictions
    gbc = joblib.load(name)
    sig_prob = gbc.predict_proba(testdf[data_cols[1:]])[:, 1]

    # compute the test statistics
    fpr_u, tpr_u, thresholds_u = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_u = auc(fpr_u, tpr_u)
    roc_info = fpr_u, tpr_u, thresholds_u, auc_u
    # save the info
    model_name = 'GBC_{0}p'.format(prong)
    write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos


def predict_base_nn(prong):
    '''
    Computes the predictions for the gradient boosted decision tree.
    Inputs:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    X_testscaled, y_test, jet_mass = load_test_data_nn(prong)
    y_test = np.ravel(y_test)
    name = str(project_dir) + '/models/base_nn_{0}p.h5'.format(prong)

    # load the model and make predictions
    base_nn = load_model(name)
    sig_prob = base_nn.predict(X_testscaled, verbose=False)
    sig_prob = np.ravel(sig_prob)
    assert sig_prob.shape == y_test.shape
    # compute the test statistics
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_score = auc(fpr, tpr)
    roc_info = fpr, tpr, thresholds, auc_score
    # save the info
    model_name = 'base_nn_{0}p'.format(prong)
    write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def main(prong):
    '''
    '''
    print('Making uboost predictions')
    uboost_hist = predict_uboost(prong)

    print('Making gradient boosted decision tree predictions')
    gbc_hist = predict_gbc(prong)

    print('Making gradient base neural network')
    bnn_hist = predict_base_nn(prong)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
