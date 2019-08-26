# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/interm directory
Trains and saves the boosted decision tree from the planed data
'''
import click
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import logging
from HelperFunctions import load_data_bdt


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
@click.option('--save', default=False, type=click.Choice(['True', 'False']),
              help='How many prongs in signal jets')
def train_PCA_gbc(prong, save):
    logger = logging.getLogger(__name__)
    logger.info('Starting to train GBC for {0} prong signal on PCA data'.format(prong))
    df, y, feats = load_data_bdt(prong, set='train', scale='pca')
    train_features = feats[1:]

    X = df[train_features]

    n_estimators = 150
    GBC = GradientBoostingClassifier(max_depth=4,
                                     n_estimators=n_estimators,
                                     learning_rate=0.1)

    GBC.fit(X, y.flatten())
    logger.info('finished training GBC for {0} prong signal'.format(prong))

    if save == 'True':
        #  save pickled classifier
        model_file_name = 'models/pca_gbc_{0}p.p'.format(prong)
        dump(GBC, model_file_name)
        logger.info('saved PCA GBC model to ' + model_file_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    train_PCA_gbc()
