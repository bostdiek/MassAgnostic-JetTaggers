# -*- coding: utf-8 -*-
"""
Author: Layne Bradshaw (layne.bradsh@gmail.com);
        Bryan Ostdiek (bostdiek@gmail.com)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson, norm
from scipy.special import loggamma
import sys


def my_log_pois(seen, expected, back_weight):
    # numerator = np.exp(-expected) * np.power(expected, seen)
    if expected == 0:
        expected = 0.5 * back_weight
    numerator = -expected + (seen * np.log(expected))

    if np.isinf(numerator):
        print(expected)
        sys.exit('bad')

    denominator = loggamma(seen + 1)

    return numerator - denominator


# start by defining likelihood functions
def unconstrained_L(params, s_vals, b_vals, dev, back_weight):
    """
    Likelihood function to be used for finding maximum likelihood estimators
    """
    mu = params[0]
    d = params[1]

    L = norm.logpdf(d, 0, dev)
    for s, b in zip(s_vals, b_vals):
        bexp = b
        if b == 0:
            bexp = 0.5 * back_weight
        if s == 0 and b == 0:
            continue
        L += my_log_pois(s + b, mu * s + (bexp * (1 + d)), back_weight)
        # L += poisson.logpmf(round(s + b), mu * s + (b * (1 + d)))
    return -2 * L


def constrained_L(d, mu, s_vals, b_vals, dev, back_weight):
    """
    Likelihood function to be used for finding the conditional maximum
    likelihood estimator for d, given a fixed value of mu (taken to be 0)
    """
    L = norm.logpdf(d, 0, dev)
    for s, b in zip(s_vals, b_vals):
        if b == 0:
            bexp = 0.5 * back_weight
        else:
            bexp = b
        if s == 0 and b == 0:
            continue
        L += my_log_pois(s + b, mu * s + (bexp * (1 + d)), back_weight)
        # L += poisson.logpmf(round(s + b), mu * s + (b * (1 + d)))
    return -2 * L


def get_q0_hist(sig_hist, back_hist, uncert, back_weight):
    '''
    Computes q0 for a given efficiency.
    Inputs:
        sig_hist: histogram of the number of signal in each bin
        back_hist: histogram of the number of background in each bin
            histograms must be the same length.
        uncert: (float) the size of the uncertainty on the background normalization.
            For example, 0.5 would correspond to a 50% uncertainty
    Outputs:
        q0: (float) the test statistic.
    '''
    sig = sig_hist
    back = back_hist

    assert len(sig) == len(back)
    if np.sum(back) == 0:
        return 0
    L_unconstrained_max = minimize(unconstrained_L,
                                   args=(sig, back, uncert, back_weight),
                                   x0=[5, 0.1],
                                   # bounds=((0, None),
                                   #         (None, None)
                                   #         ),
                                   method='Nelder-Mead'
                                   )

    L_constrained_max = minimize(constrained_L,
                                 args=(0, sig, back, uncert, back_weight),
                                 x0=[0.1],
                                 # bounds=((0, None))
                                 method='Nelder-Mead'
                                 )
    q0 = L_constrained_max.fun - L_unconstrained_max.fun
    if L_unconstrained_max.x[0] < 0:
        q0 = 0
    return q0


def get_test_stat_from_hist_dict(dictionary, method, eff, uncert, bin_size):
    '''
    Computes sqrt(q0) for a given efficiency.
    Inputs:
        dictionary: dictionary should keys of the different efficiencies.
            The signal masses are in element 0
            The background masses are in element 1
        eff: (float) the signal efficiency
        uncert: (float) the size of the uncertainty on the background normalization.
            For example, 0.5 would correspond to a 50% uncertainty
        bin_size: size of the bins for the jet mass
    Outputs:
        sqrt_q0: (float) the sqrt of the test statistic.
    '''
    sig = dictionary[method][eff][0]
    back = dictionary[method][eff][1]

    s_weight = float(100) / float(dictionary[method][1][0].shape[0])
    b_weight = float(10000) / float(dictionary[method][1][1].shape[0])

    sig_hist = np.histogram(sig, bins=np.linspace(50, 400, (400 - 50) / bin_size),
                            weights=s_weight * np.ones_like((sig)))
    back_hist = np.histogram(back, bins=np.linspace(50, 400, (400 - 50) / bin_size),
                             weights=b_weight * np.ones_like((back)))

    s_vals = np.ndarray.tolist(sig_hist[0])
    b_vals = np.ndarray.tolist(back_hist[0])
    assert len(s_vals) == len(b_vals)

    L_unconstrained_max = minimize(unconstrained_L,
                                   args=(s_vals, b_vals, uncert),
                                   x0=[1, 1],
                                   method='Nelder-Mead'
                                   )
    L_constrained_max = minimize(constrained_L,
                                 args=(0, s_vals, b_vals, uncert),
                                 x0=[1],
                                 method='Nelder-Mead'
                                 )
    sqrt_q0 = np.sqrt(L_constrained_max.fun - L_unconstrained_max.fun)
    return sqrt_q0
