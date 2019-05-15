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

datadir = 'data/interim/'

data_cols = ['(pruned)m', 'pT', 'scaled_tau_(1)^(1/2)',
             'scaled_tau_(1)^(1)', 'scaled_tau_(1)^(2)',
             'scaled_tau_(2)^(1/2)', 'scaled_tau_(2)^(1)', 'scaled_tau_(2)^(2)',
             'scaled_tau_(3)^(1/2)', 'scaled_tau_(3)^(1)', 'scaled_tau_(3)^(2)',
             'scaled_tau_(4)^(1)', 'scaled_tau_(4)^(2)'
             ]


def load_data(prong):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Outputs:
        df: pandas dataframe with the scaled data and mass (to flatten)
        y: np.array of training labels
        features: mass + names of columns to train on
    '''
    #  Training data
    train_x_name = 'train_scaled_X_{0}p.npy'.format(prong)
    X_trainscaled = np.load(datadir + train_x_name)

    y_train_name = 'train_Y_{0}p.npy'.format(prong)
    y = np.load(datadir + y_train_name)

    mass_name = 'train_jetmass_{0}p.npy'.format(prong)
    mass = np.load(datadir + mass_name).reshape(-1, 1)

    #  Combine mass and features
    X = np.hstack((mass, X_trainscaled))
    df = pd.DataFrame(X,
                      columns=data_cols
                      )

    return df, y, data_cols


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
    df, y, feats = load_data(prong)
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
