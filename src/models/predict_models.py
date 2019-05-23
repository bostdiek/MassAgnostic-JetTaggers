# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/interm directory
Loads models previously saved
Makes predictions and saves them to the modelpredictions directory
'''
import click
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
import pickle
from pathlib import Path
import logging

from HelperFunctions import load_data_bdt


def predict_bdts(prong):
    '''
    '''
    testdf, y_test, data_cols = load_data_bdt(prong, set='test')
    name = str(project_dir) + '/models/uBoost_{0}p.p'.format(prong)
    uboost = joblib.load(name)


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def main(prong):
    '''
    '''
    predict_bdts(2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
