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
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import logging
from HelperFunctions import load_data_bdt


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
@click.option('--save', default='False', type=click.Choice(['True', 'False']),
              help='How many prongs in signal jets')
def train_GBC(prong, save):
    '''
    Input:
        prong: interger denoting the number of prongs in the signal jets
    Output:
        none
    '''
    logger = logging.getLogger(__name__)
    logger.info('Starting to train GBC for {0} prong signal'.format(prong))
    df, y, feats = load_data_bdt(prong)
    train_features = feats[1:]
    print('Training on, ', train_features)

    X = df[train_features]

    n_estimators = 150
    GBC = GradientBoostingClassifier(max_depth=4,
                                     n_estimators=n_estimators,
                                     learning_rate=0.1)

    GBC.fit(X, y.flatten())
    logger.info('finished training GBC for {0} prong signal'.format(prong))

    #  save pickled classifier
    if save == 'True':
        model_file_name = 'models/GBC_{0}p.p'.format(prong)
        dump(GBC, model_file_name)
        logger.info('saved GBC model to ' + model_file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    train_GBC()
