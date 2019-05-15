# -*- coding: utf-8 -*-
'''
Authors: Rashmish Mishra
         Bryan Ostdiek (bostdiek@gmail.com)

This file will reads data from the data/interm directory
Trains and saves a uBoost classifier
'''
import click
import numpy as np
import pandas as pd
from hep_ml import uboost
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import logging
from HelperFunctions import load_data_bdt


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def train_uBoost(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Output:
        none
    '''
    logger = logging.getLogger(__name__)
    logger.info('Starting to train uBoost for {0} prong signal'.format(prong))
    df, y, feats = load_data_bdt(prong)
    uniform_features_mass = [feats[0]]
    train_features = feats[1:]
    print('trying to keep ' + uniform_features_mass[0] + ' flat')

    n_estimators = 150
    base_estimator = DecisionTreeClassifier(max_depth=4)

    uboost_clf = uboost.uBoostClassifier(uniform_features=uniform_features_mass,
                                         uniform_label=0,  # flatten the background
                                         base_estimator=base_estimator,
                                         train_features=train_features,
                                         n_estimators=n_estimators,
                                         n_threads=4,
                                         # efficiency_steps=12,
                                         )
    uboost_clf.fit(df, y.flatten())
    logger.info('finished training uBoost for {0} prong signal'.format(prong))

    #  save pickled classifier
    model_file_name = 'models/uBoost_{0}p.p'.format(prong)
    dump(uboost_clf, model_file_name)
    logger.info('saved uBoost model to ' + model_file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    train_uBoost()
