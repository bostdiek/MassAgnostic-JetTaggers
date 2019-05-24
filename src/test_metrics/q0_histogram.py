"""
Author: Layne Bradshaw (layne.bradsh@gmail.com)

"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson, norm
import pickle

# start by defining likelihood functions

def unconstrained_L(params,s_vals,b_vals,dev):
    """
    Likelihood function to be used for finding maximum likelihood estimators
    """
    mu = params[0]
    d = params[1]

    L = norm.logpdf(d,0,dev)
    for s,b in zip(s_vals,b_vals):
        L += poisson.logpmf(round(s+b),mu*s+(b*(1+d)))
    return -2*L

def constrained_L(d,mu,s_vals,b_vals,dev):
    """
    Likelihood function to be used for finding the conditional maximum
    likelihood estimator for d, given a fixed value of mu (taken to be 0)
    """

    L = norm.logpdf(d,0,dev)
    for s,b in zip(s_vals,b_vals):
        L += poisson.logpmf(round(s+b),mu*s+(b*(1+d)))
    return -2*L

def get_test_stat_from_pickle(dictionary,method,eff,uncert,bin_size):

    sig = dictionary[method][eff][0]
    back = dictionary[method][eff][1]

    s_weight = float(100)/float(dictionary[method][1][0].shape[0])
    b_weight = float(10000)/float(dictionary[method][1][1].shape[0])

    sig_hist = np.histogram(sig,bins=np.linspace(50,400,(400-50)/bin_size),
                            weights = s_weight*np.ones_like((sig)))
    back_hist = np.histogram(back,bins=np.linspace(50,400,(400-50)/bin_size),
                             weights = b_weight*np.ones_like((back)))

    s_vals = np.ndarray.tolist(sig_hist[0])
    b_vals = np.ndarray.tolist(back_hist[0])
    assert len(s_vals)==len(b_vals)

    L_unconstrained_max = minimize(unconstrained_L,args=(s_vals,b_vals,uncert),
                                    x0=[1,1],method='Nelder-Mead')
    L_constrained_max = minimize(constrained_L,args=(0,s_vals,b_vals,uncert),
                                  x0=[1],method='Nelder-Mead')

    return np.sqrt(L_constrained_max.fun-L_unconstrained_max.fun)
