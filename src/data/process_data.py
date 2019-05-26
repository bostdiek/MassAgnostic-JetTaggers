# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/raw directory, splits it into
    training, validation, and test data, and scales the data.
The resulting data is saved to the data/interim directory.
'''
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PCA_scaler import PCA_scaler_withRotations
import logging
import pickle

raw_dir = 'data/raw/'
data_cols = ['(pruned)m', 'pT', 'tau_(1)^(1/2)',
             'tau_(1)^(1)', 'tau_(1)^(2)',
             'tau_(2)^(1/2)', 'tau_(2)^(1)', 'tau_(2)^(2)',
             'tau_(3)^(1/2)', 'tau_(3)^(1)', 'tau_(3)^(2)',
             'tau_(4)^(1)', 'tau_(4)^(2)'
             ]


def load_data(prong):
    '''
    loads data from the raw directory.
    input: int (prong) - if the signal file has 2, 3, or 4 prong jets
    returns: two pandas DataFrames, one signal one background
    '''
    logger = logging.getLogger(__name__)
    logger.info('Loading data for {0}p prong'.format(prong))
    SignalDF = pd.read_csv(raw_dir + 'data_sig_{}p.txt'.format(prong),
                           skiprows=2,
                           index_col=False,
                           names=data_cols
                           )
    BackgroundDF = pd.read_csv(raw_dir + 'data_bkg.txt',
                               skiprows=2,
                               index_col=False,
                               names=data_cols
                               )
    return SignalDF, BackgroundDF


def process_data(prong):
    '''Goes through the data which is 2, 3, or 4 prong'''
    logger = logging.getLogger(__name__)
    SignalDF, BackgroundDF = load_data(prong)
    TrainingColumns = SignalDF.columns

    SignalDF['Label'] = 1
    BackgroundDF['Label'] = 0

    CombinedData = np.vstack([SignalDF[TrainingColumns].values,
                              BackgroundDF[TrainingColumns].values
                              ]
                             )
    CombinedLabels = np.hstack([SignalDF['Label'].values,
                                BackgroundDF['Label'].values
                                ]
                               ).reshape(CombinedData.shape[0], 1)
    print(CombinedData.shape)

    if not os.path.isfile('data/interim/TrainingIndices_{0}p.npy'.format(prong)):
        indices = np.arange(CombinedLabels.shape[0])
        np.random.shuffle(indices)
        training_size = int(0.7 * len(indices))
        validation_size = int(0.15 * len(indices))
        TrainingIndices = indices[: training_size]
        print(TrainingIndices.shape)
        ValIndices = indices[training_size: training_size + validation_size]
        TestIndices = indices[training_size + validation_size:]
        print(ValIndices.shape)
        print(TestIndices.shape)
        np.save('data/interim/TrainingIndices_{0}p.npy'.format(prong), TrainingIndices)
        np.save('data/interim/ValidationIndices_{0}p.npy'.format(prong), ValIndices)
        np.save('data/interim/TestIndices_{0}p.npy'.format(prong), TestIndices)
    else:
        TrainingIndices = np.load('data/interim/TrainingIndices_{0}p.npy'.format(prong))
        ValIndices = np.load('data/interim/ValidationIndices_{0}p.npy'.format(prong))
        TestIndices = np.load('data/interim/TestIndices_{0}p.npy'.format(prong))

    X_train, y_train = CombinedData[TrainingIndices], CombinedLabels[TrainingIndices]
    X_test, y_test = CombinedData[TestIndices], CombinedLabels[TestIndices]
    X_val, y_val = CombinedData[ValIndices], CombinedLabels[ValIndices]

    mass_train = X_train[:, 0]
    mass_test = X_test[:, 0]
    mass_val = X_val[:, 0]
    np.save('data/interim/train_jetmass_{0}p.npy'.format(prong),
            mass_train
            )
    np.save('data/interim/test_jetmass_{0}p.npy'.format(prong),
            mass_test
            )
    np.save('data/interim/val_jetmass_{0}p.npy'.format(prong),
            mass_val
            )

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    X_val = X_val[:, 1:]
    np.save('data/interim/train_X_{0}p.npy'.format(prong),
            X_train
            )
    np.save('data/interim/test_X_{0}p.npy'.format(prong),
            X_test
            )
    np.save('data/interim/val_X_{0}p.npy'.format(prong),
            X_val
            )

    class_weights = {1: float(len(X_train)) / np.sum(y_train == 1),
                     0: float(len(X_train)) / np.sum(y_train == 0)
                     }
    print(class_weights)
    with open('data/interim/class_weights_{0}p.p'.format(prong), 'wb') as cf:
        pickle.dump(class_weights, cf)

    val_weights = np.ones_like(y_val)
    val_weights[y_val == 0] = class_weights[0]
    val_weights[y_val == 1] = class_weights[1]
    val_weights = val_weights.flatten()
    np.save('data/interim/val_class_weights_indiv_{0}p.npy'.format(prong),
            val_weights
            )

    tr_weights = np.ones_like(y_train)
    tr_weights[y_train == 0] = class_weights[0]
    tr_weights[y_train == 1] = class_weights[1]
    tr_weights = tr_weights.flatten()
    np.save('data/interim/train_class_weights_indiv_{0}p.npy'.format(prong),
            tr_weights
            )

    SS = StandardScaler()
    X_trainscaled = SS.fit_transform(X_train)
    X_testscaled = SS.transform(X_test)
    X_valscaled = SS.transform(X_val)

    logger.info('saving scaled and split data')
    np.save('data/interim/train_scaled_X_{0}p.npy'.format(prong),
            X_trainscaled
            )
    np.save('data/interim/test_scaled_X_{0}p.npy'.format(prong),
            X_testscaled
            )
    np.save('data/interim/val_scaled_X_{0}p.npy'.format(prong),
            X_valscaled
            )
    np.save('data/interim/train_Y_{0}p.npy'.format(prong),
            y_train
            )
    np.save('data/interim/test_Y_{0}p.npy'.format(prong),
            y_test
            )
    np.save('data/interim/val_Y_{0}p.npy'.format(prong),
            y_val
            )


def DoPCA(prong):
    SignalDF, BackgroundDF = load_data(prong)
    TrainingColumns = SignalDF.columns

    SignalDF['Label'] = 1
    BackgroundDF['Label'] = 0

    CombinedData = np.vstack([SignalDF[TrainingColumns].values,
                              BackgroundDF[TrainingColumns].values
                              ]
                             )
    CombinedLabels = np.hstack([SignalDF['Label'].values,
                                BackgroundDF['Label'].values
                                ]
                               ).reshape(CombinedData.shape[0], 1)
    TrainingIndices = np.load('data/interim/TrainingIndices_{0}p.npy'.format(prong))
    ValIndices = np.load('data/interim/ValidationIndices_{0}p.npy'.format(prong))
    TestIndices = np.load('data/interim/TestIndices_{0}p.npy'.format(prong))

    X_train, y_train = CombinedData[TrainingIndices], CombinedLabels[TrainingIndices]
    X_test, y_test = CombinedData[TestIndices], CombinedLabels[TestIndices]
    X_val, y_val = CombinedData[ValIndices], CombinedLabels[ValIndices]

    mass_train = X_train[:, 0]
    mass_test = X_test[:, 0]
    mass_val = X_val[:, 0]
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    X_val = X_val[:, 1:]

    bkg_data_train = X_train[np.ravel(y_train) == 0]
    bkg_mass_train = mass_train[np.ravel(y_train) == 0]

    MyPCA = PCA_scaler_withRotations(jet_mass=bkg_mass_train,
                                     data=bkg_data_train,
                                     bins=100,
                                     minmass=50,
                                     maxmass=400)
    PCA_train = MyPCA.transform(jet_mass=mass_train, data=X_train)
    PCA_val = MyPCA.transform(jet_mass=mass_val, data=X_val)
    PCA_test = MyPCA.transform(jet_mass=mass_test, data=X_test)

    np.save('data/interim/train_X_PCA_{0}p.npy'.format(prong), PCA_train)
    np.save('data/interim/test_X_PCA_{0}p.npy'.format(prong), PCA_test)
    np.save('data/interim/val_X_PCA_{0}p.npy'.format(prong), PCA_val)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
