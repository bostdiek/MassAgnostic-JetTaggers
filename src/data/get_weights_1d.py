# -*- coding: utf-8 -*-
'''
Author: Layne Bradshaw
Computes the paning weights to flatten the mass distribtion
'''
import numpy as np
import pandas as pd
from scipy import interpolate


def load_data(prong):
    '''
    loads data from the interim directory for the signal with n-prong jets
    returns numpy arrays for the jet mass, along with the signal/background label
    '''
    datadir = 'data/interim/'
    m_train = 'train_jetmass_{0}p.npy'.format(prong)
    m_test = 'test_jetmass_{0}p.npy'.format(prong)
    m_val = 'val_jetmass_{0}p.npy'.format(prong)

    y_train = 'train_Y_{0}p.npy'.format(prong)
    y_test = 'test_Y_{0}p.npy'.format(prong)
    y_val = 'val_Y_{0}p.npy'.format(prong)

    data = ((np.load(datadir + m_train), np.load(datadir + y_train)),
            (np.load(datadir + m_test), np.load(datadir + y_test)),
            (np.load(datadir + m_val), np.load(datadir + y_val))
            )
    return data


def get_probs_1d(x, min_val, max_val, bins=100):
    """
    Inputs:
    x - data to plane (must have shape (x.shape[0],) )
    min_val - minimum value to be considered. Since we use the midpoint of each bin,
              set this value to something
              lower than the minimum of the dataset

    max_val - maximum value to be considered. Since we use the midpoint of each bin,
              set this value to something
              higher than the maximum value of the dataset

    bins - number of bins to use when planing

    Outputs:

    interp - interpolating function which returns the normalized probability
    """

    nbins = (np.linspace(min_val, max_val, bins),)
    xpts = 0.5 * (nbins[0][1:] + nbins[0][:-1])

    vals, _ = np.histogram(x, bins=nbins[0], density=True)
    interp_pts = np.meshgrid(xpts, indexing='ij', sparse=True)
    interp = interpolate.interp1d(interp_pts[0], vals, bounds_error=False)
    return interp


def get_weights(x, interp):
    '''
    Calculates the normalized total weights
    Inputs:
        x - data for which to calculate weigths
        interp - interpolating function for the probability of a given mass
    Outputs:
        weights - np.array of floats used to flatten a distribution
    '''
    probs = interp(x)  # get value of each bin with interpolating function
    weights = probs.sum() / probs  # normalize weights
    weights /= weights.mean()  # normalize weights
    weights = weights.reshape(-1, 1)
    return weights


def set_weights(prong):
    data = load_data(prong)
    m_train, y_train = data[0]
    m_test, y_test = data[1]
    m_val, y_val = data[2]

    min_mass, max_mass = np.min(m_train), np.max(m_train)
    #  set the used min and max to 5% of the difference larger and smaller
    diff = max_mass - min_mass
    min_mass, max_mass = min_mass - 0.05 * diff, max_mass + 0.05 * diff

    sig_mass = m_train[y_train == 1]
    sig_probs = get_probs_1d(x=sig_mass,
                             min_val=min_mass,
                             max_val=max_mass,
                             bins=100)

    back_mass = m_train[y_train == 0]
    back_probs = get_probs_1d(x=sig_mass,
                              min_val=min_mass,
                              max_val=max_mass,
                              bins=100)

    # new array for the output weights
    train_weights = np.zeros_like


print('getting weights')
sig_weights = get_weights_1d(sig_mass,min_val=75,max_val=325,bins=100)
back_weights = get_weights_1d(back_mass,min_val=75,max_val=325,bins=100)
print('done getting weights')
