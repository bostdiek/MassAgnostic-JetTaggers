# -*- coding: utf-8 -*-
"""
Distance.py
Author: Bryan Ostdiek (bostdiek@gmail.com)
Computes the Bhattacharyya distance metric
"""

import numpy as np
from scipy.stats import entropy


def BhatDist(hist_1, hist_2):
    '''
    Computes the Bhattacharyya distance metric
    Inputs:
        hist_1: (np.array) values for the bins of jet mass at an efficiency
        hist_2: (np.array) values for the bins of jet mass at a different efficiency
    The histograms need to be the same length

    Outputs:
        dist: (float) the Bhattacharyya distance
    '''
    assert len(hist_1) == len(hist_2)
    # normalize
    hist_1 = hist_1 / np.sum(hist_1)
    hist_2 = hist_2 / np.sum(hist_2)
    h1_ = np.mean(hist_1)
    h2_ = np.mean(hist_2)
    N = len(hist_1)

    right = np.sum(np.sqrt(hist_1 * hist_2)) / np.sqrt(h1_ * h2_ * N**2)
    if right <= 1:
        db = np.sqrt(1 - right)
    else:
        db = 0
    # dist_e = -np.log(np.sum(np.sqrt(hist_1 * hist_2)))

    return db  # , dist_e


def JS_Distance(hist_1, hist_2):
    '''
    Inputs:
        hist_1: before cuts
        hist_2: after cuts
    Outputs:
        Jensen-Shannon distance
    '''
    assert len(hist_1) == len(hist_2)

    # Normalize
    hist_1 /= np.sum(hist_1)
    hist_2 /= np.sum(hist_2)
    m = (hist_1 + hist_2) / 2

    js = np.sqrt((entropy(hist_1, m) + entropy(hist_2, m)) / 2)
    return js
