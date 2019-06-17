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
import sys

from HelperFunctions import load_data_bdt, load_test_data_nn
from HelperFunctions import write_roc_pickle, make_histos
from HelperFunctions import load_test_data_PCA_nn


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
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


def predict_gbc(prong, data='base'):
    '''
    Computes the predictions for the gradient boosted decision tree.
    Inputs:
        prong: interger denoting the number of prongs in the signal jets
        data: options are 'base', 'planed', 'pca' indicating which dataset
            the gbc was trained on
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    if data == 'base':
        prefix = ''
        scale = 'normal'
    elif data == 'planed':
        prefix = 'planed'
        scale = 'normal'
    elif data == 'pca':
        prefix = 'pca'
        scale = 'pca'
    else:
        sys.exit('Bad entry for data type')

    testdf, y_test, data_cols = load_data_bdt(prong, set='test', scale=scale)
    jet_mass = testdf[data_cols[0]].values
    if prefix == '':
        name = str(project_dir) + '/models/GBC_{0}p.p'.format(prong)
    else:
        name = str(project_dir) + '/models/{0}_gbc_{1}p.p'.format(prefix, prong)

    # load the model and make predictions
    gbc = joblib.load(name)
    sig_prob = gbc.predict_proba(testdf[data_cols[1:]])[:, 1]

    # compute the test statistics
    fpr_u, tpr_u, thresholds_u = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_u = auc(fpr_u, tpr_u)
    roc_info = fpr_u, tpr_u, thresholds_u, auc_u
    # save the info
    if prefix == '':
        model_name = 'GBC_{0}p'.format(prong)
    else:
        model_name = '{0}_GBC_{1}p'.format(prefix, prong)
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


def predict_base_nn(prong):
    '''
    Computes the predictions for the base neural network.
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
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


def predict_planed_nn(prong):
    '''
    Computes the predictions for the neural network trained on planed data.
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
    name = str(project_dir) + '/models/planed_nn_{0}p.h5'.format(prong)

    # load the model and make predictions
    planed_nn = load_model(name)
    sig_prob = planed_nn.predict(X_testscaled, verbose=False)
    sig_prob = np.ravel(sig_prob)
    assert sig_prob.shape == y_test.shape
    # compute the test statistics
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_score = auc(fpr, tpr)
    roc_info = fpr, tpr, thresholds, auc_score
    # save the info
    model_name = 'planed_nn_{0}p'.format(prong)
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


def predict_ann(prong, lam_exp):
    '''
    Computes the predictions for the adversarial neural network.
    Inputs:
        prong: [interger] denoting the number of prongs in the signal jets
        lam_exp: [string] the adversary lambda is given by 10**(lam_exp)
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    X_testscaled, y_test, jet_mass = load_test_data_nn(prong)
    y_test = np.ravel(y_test)
    name = str(project_dir) + '/models/nn_with_adv_lam_{0}'.format(lam_exp)
    name += '_final_{0}p.h5'.format(prong)

    # load the model and make predictions
    model = load_model(name)
    sig_prob = model.predict(X_testscaled, verbose=False)
    sig_prob = np.ravel(sig_prob)
    assert sig_prob.shape == y_test.shape

    # compute the test statistics
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_score = auc(fpr, tpr)
    roc_info = fpr, tpr, thresholds, auc_score

    # save the info
    lam = 10**float(lam_exp)
    if lam > 0:
        lam = round(lam)
    model_name = 'adv_nn_lam_{0:03d}_{1}p'.format(lam, prong)
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


def predict_PCA_nn(prong):
    '''
    Computes the predictions for the base neural network.
    Inputs:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        sig_prob: [np array] probability of being selected for each event
        fpr: [np array] false positive rate
        tpr: [np array] true postive rate
        threholds: [np array] cut values corresponding to the fpr and tpr
        auc: [float] the area under the ROC curve
    '''
    X_testscaled, y_test, jet_mass = load_test_data_PCA_nn(prong)
    y_test = np.ravel(y_test)
    name = str(project_dir) + '/models/pca_nn_{0}p.h5'.format(prong)

    # load the model and make predictions
    model = load_model(name)
    sig_prob = model.predict(X_testscaled, verbose=False)
    sig_prob = np.ravel(sig_prob)
    assert sig_prob.shape == y_test.shape
    # compute the test statistics
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=sig_prob)
    auc_score = auc(fpr, tpr)
    roc_info = fpr, tpr, thresholds, auc_score
    # save the info
    model_name = 'pca_nn_{0}p'.format(prong)
    roc_curve_out = write_roc_pickle(model_name, roc_info)
    histos = make_histos(model_name, jet_mass, sig_prob, y_test, roc_info)

    return histos, roc_curve_out


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def main(prong):
    '''
    '''
    HistDictionary = {}
    ROCDictionary = {}

    print('Making gradient boosted decision tree predictions')
    gbc_hist, gbc_roc = predict_gbc(prong, data='base')
    HistDictionary['GradientBoostingClassifier'] = gbc_hist
    ROCDictionary['GradientBoostingClassifier'] = gbc_roc

    print('Making base neural network')
    bnn_hist, bnn_roc = predict_base_nn(prong)
    HistDictionary['BaseNeuralNetwork'] = bnn_hist
    ROCDictionary['BaseNeuralNetwork'] = bnn_roc

    print('Making planed neural network')
    pnn_hist, pnn_roc = predict_planed_nn(prong)
    HistDictionary['PlanedNeuralNetwork'] = pnn_hist
    ROCDictionary['PlanedNeuralNetwork'] = pnn_roc

    print('Making planed GBC')
    pgbc_hist, pgbc_roc = predict_gbc(prong, data='planed')
    HistDictionary['PlanedGBC'] = pgbc_hist
    ROCDictionary['PlanedGBC'] = pgbc_roc

    print('Making PCA neural network')
    bnn_hist, bnn_roc = predict_PCA_nn(prong)
    HistDictionary['PCANeuralNetwork'] = bnn_hist
    ROCDictionary['PCANeuralNetwork'] = bnn_roc

    print('Making PCA GBC')
    pcagbc_hist, pcagbc_roc = predict_gbc(prong, data='pca')
    HistDictionary['PCAGBC'] = pcagbc_hist
    ROCDictionary['PCAGBC'] = pcagbc_roc

    lam_exp_list = ['0', '3.010e-01', '6.990e-01',
                    '1', '1.301e+00', '1.699e+00',
                    '2', '2.301e+00', '2.699e+00', '3']

    for le in lam_exp_list:
        lam = 10**float(le)
        if lam > 0:
            lam = round(lam)
        print('Making adversarial predictions for lambda={0:03d}'.format(lam))
        model_name = 'AdversaryLambda_{0:03d}'.format(lam)
        ann_hist, ann_roc = predict_ann(prong, le)
        HistDictionary[model_name] = ann_hist
        ROCDictionary[model_name] = ann_roc

    print('Making uboost predictions')
    uboost_hist, uboost_roc = predict_uboost(prong)
    HistDictionary['uBoost'] = uboost_hist
    ROCDictionary['uBoost'] = uboost_roc

    pred_datadir = str(project_dir.resolve()) + '/data/modelpredictions/'
    with open(pred_datadir + 'Histograms_{0}p.p'.format(prong), 'wb') as f:
        pickle.dump(HistDictionary, f)

    with open(pred_datadir + 'ROCCurves_{0}p.p'.format(prong), 'wb') as f:
        pickle.dump(ROCDictionary, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
