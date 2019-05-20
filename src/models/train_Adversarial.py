# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will reads data from the data/interm directory
Trains and saves the adversary along with the classifier
'''
import click

import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, SGD, RMSprop

import logging
import numpy as np
import os
import sys
# from pathlib import Path
import pickle

from HelperFunctions import load_data_nn

import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1,
                              patience=5, min_lr=1.0e-6)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')


def set_mass_bins(m_train, m_val, y_train):
    '''
    Digitizes and sets the mass bins.
    Inputs:
        m_train: np array of jet masses for training data
        m_val: no array of jet masses for validation data
        y_train: labels for training data
    Outputs:
        mbin_train_labels: categorical labels for keras for mass bins
        mbin_val_labels: categorical labels for keras for mass bins
    '''
    #  use the background distribution to set the bins
    mass_bins_setup = m_train[np.ravel(y_train == 0)]
    mass_bins_setup.sort()
    size = int(len(mass_bins_setup) / 10)

    massbins = [50-1e-5,
                mass_bins_setup[size], mass_bins_setup[size * 2],
                mass_bins_setup[size * 3], mass_bins_setup[size * 4],
                mass_bins_setup[size * 5], mass_bins_setup[size * 6],
                mass_bins_setup[size * 7], mass_bins_setup[size * 8],
                mass_bins_setup[size * 9],
                400+1e-5]
    print(massbins)
    #  get which bin each mass point is if __name__ == '__main__':
    mbin_train = np.digitize(m_train, massbins) - 1
    mbin_val = np.digitize(m_val, massbins) - 1

    mbin_train_labels = keras.utils.to_categorical(mbin_train, num_classes=10)
    mbin_val_labels = keras.utils.to_categorical(mbin_val, num_classes=10)

    return mbin_train_labels, mbin_val_labels


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
@click.option('--lam', default=0, type=click.IntRange(0, 12),
              help='Exponential power of lambda')
def train_Adversary(prong, lam):
    logger = logging.getLogger(__name__)
    # lam_exp -= 2
    # lam = 10 ** lam_exp
    lam_exp = '{0:.03e}'.format(np.log10(lam))
    logger.info('Working on lambda={0:0.1e}'.format(lam))
    data = load_data_nn(prong)
    X_trainscaled = data[0]
    y_train = data[1]
    X_valscaled = data[2]
    y_val = data[3]
    val_weights = data[4]
    class_weights = data[5]

    tr_weights = np.load('data/interim/train_class_weights_indiv_{0}p.npy'.format(prong))

    m_train = np.load('data/interim/train_jetmass_{0}p.npy'.format(prong))
    m_val = np.load('data/interim/val_jetmass_{0}p.npy'.format(prong))
    mbin_train_labels, mbin_val_labels = set_mass_bins(m_train, m_val, y_train)

    classifier_name = 'models/base_nn_{0}p.h5'.format(prong)
    if not os.path.isfile(classifier_name):
        logger.error('Need to run base neural networks first')
        sys.exit(classifier_name + ' does not exist')
    else:
        inputs = Input(shape=(X_trainscaled.shape[1], ))
        Classifier = Dense(50, activation='relu')(inputs)
        Classifier = Dense(50, activation='relu')(Classifier)
        Classifier = Dense(50, activation='relu')(Classifier)
        Classifier = Dense(1, activation='sigmoid')(Classifier)
        ClassifierModel = Model(inputs=inputs, outputs=Classifier)
        ClassifierModel.compile(optimizer='adam', loss='binary_crossentropy')
        ClassifierModel.summary()
        # history = ClassifierModel.fit(X_trainscaled,
        #                               y_train,
        #                               validation_data=[X_valscaled, y_val, val_weights],
        #                               epochs=100,
        #                               class_weight=class_weights,
        #                               callbacks=[reduce_lr, es]
        #                               )
        ClassifierModel = load_model('models/base_nn_{0}p.h5'.format(prong))
        ClassifierModel.name = 'Classifier'

        # with open('models/histories/base_nn_hist_{0}p.p'.format(prong), 'wb') as f:
        #     pickle.dump(history, f)

    # ***************************************************
    # Add in the Adversary
    # Now the adversary uses the whole input, but only takes the output of the classifier
    # ***************************************************
    Adversary = ClassifierModel(inputs)
    Adversary = Dense(50, activation='tanh', kernel_initializer='normal')(Adversary)
    Adversary = Dense(50, activation='tanh', kernel_initializer='normal')(Adversary)
    Adversary = Dense(10, activation='softmax', kernel_initializer='normal')(Adversary)

    # The adversary only is supposed to work on the backround events
    # feed into it the actual label, so that we can make the loss function be 0
    # for the signal events
    LabelWeights = Input(shape=(1,))
    AdversaryC = concatenate([Adversary, LabelWeights], axis=1)

    AdversaryModel = Model(inputs=[inputs, LabelWeights],
                           outputs=AdversaryC
                           )

    def Make_loss_A(alam):
        def loss(y_true, y_pred):
            y_pred, l_true = y_pred[:, :-1], y_pred[:, -1]  # prediction and label
            # y_pred = K.clip(y_pred, K.epsilon, 1 - K.epsilon)
            l_true = K.reshape(l_true, (-1, 1))
            loss_sum = K.sum(alam * -(y_true * K.log(y_pred)) * (1 - l_true), axis=-1)
            return K.mean(loss_sum)  # / K.sum(1 - l_true, axis=0)
        return loss

    CombinedLoss = Make_loss_A(-lam)
    AdvLoss = Make_loss_A(1.0)

    AdversaryModel.compile(loss=AdvLoss,
                           optimizer=Adam()
                           )
    # ***************************************************
    # Let the adversary learn for a while
    # ***************************************************
    mdir = 'models/adv/'.format(lam, prong)
    if not os.path.isdir(mdir):
        os.mkdir(mdir)
    adv_name = 'models/adv/initial_AdversaryTanhAdam_{0}p.h5'.format(prong)
    if not os.path.isfile(adv_name):
        ClassifierModel.trainable = False
        AdversaryModel.compile(loss=AdvLoss,
                               optimizer=Adam()
                               )
        AdversaryModel.summary()
        # print(np.sum(y_train == 0), np.sum(y_train == 1))
        # print(np.sum(y_val == 0), np.sum(y_val == 1))
        #
        # preds = AdversaryModel.predict([X_trainscaled, y_train], batch_size=1000000)
        # preds, l_true = preds[:, :-1], preds[:, -1]
        # print(l_true)
        # cce = -np.sum(np.log(preds) * mbin_train_labels * (1 - l_true).reshape(-1, 1), axis=-1)
        # # cce_b = cce * (1 - l_true)
        # cce_sc = cce / np.sum(1 - l_true)
        # print(preds)
        # print(cce)
        # # print(cce_b)
        # print(np.sum(cce))
        # print(cce_sc)
        # print(np.mean(cce_sc[:32]))
        # print(np.mean(cce_sc))
        # print(len(X_trainscaled[::64]))
        # print(len(X_trainscaled[::64]) * np.mean(cce_sc))
        # print(AdversaryModel.evaluate(x=[X_trainscaled, y_train],
        #                               y=mbin_train_labels,
        #                               batch_size=32,
        #                               verbose=0
        #                               ))
        # print('validation')
        # preds = AdversaryModel.predict([X_valscaled, y_val], batch_size=1000000)
        # preds, l_true = preds[:, :-1], preds[:, -1]
        # print(l_true)
        # cce = -np.sum(np.log(preds) * mbin_val_labels * (1 - l_true).reshape(-1, 1), axis=-1)
        # # cce_b = cce * (1 - l_true)
        # cce_sc = cce / np.sum(1 - l_true)
        # print(preds)
        # print(cce)
        # # print(cce_b)
        # print(np.sum(cce))
        # print(cce_sc)
        # print(np.mean(cce_sc[:32]))
        # print(np.mean(cce_sc))
        # print(len(X_trainscaled[::64]))
        # print(len(X_trainscaled[::64]) * np.mean(cce_sc))
        # print(AdversaryModel.evaluate(x=[X_valscaled, y_val],
        #                               y=mbin_val_labels,
        #                               batch_size=64,
        #                               verbose=0
        #                               ))
        # sys.exit('Full break')
        AdversaryModel.fit(x=[X_trainscaled, y_train],
                           y=mbin_train_labels,
                           validation_data=[[X_valscaled, y_val],
                                            mbin_val_labels],
                           epochs=50,
                           # callbacks=[reduce_lr, es]
                           )

        AdversaryModel.save_weights(adv_name)
        AdversaryModel.name = 'Adversary'

        preds = AdversaryModel.predict([X_valscaled, y_val])
        print(preds)
        print(mbin_val_labels[(y_val == 0).flatten()])
        print(np.argmax(mbin_val_labels[(y_val == 0).flatten()], axis=-1))
        plt.figure(figsize=(4, 4))
        # plt.hist(np.argmax(mbin_val_labels[(y_val == 0).flatten()], axis=-1),
        #          range=(1,11), bins=10)
        plt.savefig('InitialAdversaryOnValidation_{0}p.pdf'.format(prong),
                    bbox_inches='tight'
                    )
        print(np.unique(np.argmax(mbin_val_labels[(y_val == 0).flatten()], axis=-1)))
        print(np.unique(np.argmax(preds[(y_val == 0).flatten()], axis=-1)))
        plt.hist2d(np.argmax(mbin_val_labels[(y_val == 0).flatten()], axis=-1),
                   np.argmax(preds[(y_val == 0).flatten()], axis=-1),
                   bins=(range(0, 11), range(0, 11)),
                   norm=LogNorm()
                   )
        plt.xlabel('Mass (scaled)')
        plt.ylabel('Predicted')
        plt.savefig('InitialAdversaryOnValidation_{0}p.pdf'.format(prong),
                    bbox_inches='tight'
                    )
    else:
        AdversaryModel.load_weights(adv_name)
        AdversaryModel.name = 'Adversary'

    # ***************************************************
    # Now put the two models together into one model
    # With two output, there will need to be two losses
    # ***************************************************
    CombinedModel = Model(inputs=[inputs, LabelWeights],
                          outputs=[ClassifierModel(inputs),
                                   AdversaryModel([inputs, LabelWeights])
                                   ]
                          )

    losses = {"L_C": [], "L_A": [], "L_C - L_A": []}
    batch_size = 512
    min_loss = np.inf
    count = 0
    mylr = 1e-5
    ClassOpt = Adam(lr=mylr)
    AdvOpt = Adam(lr=10 * mylr)
    CombinedModel.compile(loss=['binary_crossentropy',
                                CombinedLoss],
                          optimizer=ClassOpt
                          )
    mdir = 'models/adv/lam_{0}_{1}p/'.format(lam_exp, prong)
    if not os.path.isdir(mdir):
        os.mkdir(mdir)

    #  Work through the epochs of training
    for i in range(500):
        m_losses = CombinedModel.evaluate([X_valscaled, y_val],
                                          [y_val, mbin_val_labels],
                                          sample_weight=[val_weights,
                                                         np.ones_like(val_weights)],
                                          verbose=1
                                          )
        losses["L_C - L_A"].append(m_losses[0][None][0])
        losses["L_C"].append(m_losses[1][None][0])
        losses["L_A"].append(-m_losses[2][None][0])
        print(losses["L_A"][-1] / lam)

        current_loss = m_losses[0][None][0]
        if current_loss < min_loss:
            min_loss = current_loss
            count = 0
        else:
            count += 1
        if count > 0 and count % 15 == 0:
            if mylr > 1e-5:
                mylr = mylr * np.sqrt(0.1)
                ClassOpt = Adam(lr=mylr)
                # AdvOpt = Adam(lr=mylr)
                print('Lowering learning rate to {0:1.01e}'.format(mylr))
            else:
                print('Has not improved in {1} epochs with lr={0:1.01e}'.format(mylr,
                                                                                count))
        if i % 5 == 0:
            with open('models/histories/adv_lam_{0}_{1}p.p'.format(lam_exp, prong),
                      'wb') as hf:
                pickle.dump(losses, hf)
            AdversaryModel.save_weights(mdir + 'Adv_lam_{0}_{1:03}_weigths_{2}p.h5'.format(lam_exp, i, prong))
            ClassifierModel.save_weights(mdir + 'Class_lam_{0}_{1:03}_weights_{2}p.h5'.format(lam_exp, i, prong))

        indices = np.random.permutation(len(X_trainscaled))

        # Fit Classifier
        AdversaryModel.trainable = False
        ClassifierModel.trainable = True

        CombinedModel.compile(loss=['binary_crossentropy',
                                    CombinedLoss],
                              optimizer=ClassOpt
                              )
        for j in range(5):
            indices = np.random.permutation(len(X_trainscaled))[:batch_size]
            CombinedModel.train_on_batch(x=[X_trainscaled[indices],
                                            y_train[indices]
                                            ],
                                         y=[y_train[indices],
                                            mbin_train_labels[indices]
                                            ],
                                         sample_weight=[tr_weights[indices],
                                                        np.ones_like(tr_weights[indices])
                                                        ]
                                         )

        # Fit Adversary
        AdversaryModel.trainable = True
        ClassifierModel.trainable = False
        AdversaryModel.compile(loss=AdvLoss,
                               optimizer=AdvOpt
                               )
        for j in range(200):
            indices = np.random.permutation(len(X_trainscaled))
            AdversaryModel.train_on_batch(x=[X_trainscaled[indices],
                                             y_train[indices]],
                                          y=mbin_train_labels[indices]
                                          )

    AdversaryModel.save_weights('models/Adv_lam_{0}_final_{1}p.h5'.format(lam_exp, prong))
    ClassifierModel.save('models/nn_with_adv_lam_{0}_final_{1}p.h5'.format(lam_exp, prong))
    ClassifierModel.save_weights('models/nn_with_adv_lam_{0}_final_weights_{1}p.h5'.format(lam_exp, prong))
    with open('models/histories/adv_lam_{0}_{1}p.p'.format(lam_exp, prong),
              'wb') as hf:
        pickle.dump(losses, hf)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    train_Adversary()
