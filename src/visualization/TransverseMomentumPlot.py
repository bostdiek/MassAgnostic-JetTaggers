import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from pathlib import Path
import pickle

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 12,
                     'axes.unicode_minus': False,
                     'axes.labelsize': 12,
                     'axes.labelsize': 12,
                     'figure.figsize': (4, 4),
                     'figure.dpi': 80,
                     'mathtext.fontset': 'cm',
                     'mathtext.rm': 'serif',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True
                     })

data_dir = 'data/interim/'

plt.figure(figsize=(8.5, 2.3))
gs0 = gs.GridSpec(1, 4, wspace=0.1,)
for prong in [2, 3, 4]:
    pt = np.load(data_dir + 'train_jetpT_{0}p.npy'.format(prong))
    labels = np.load(data_dir + 'train_Y_{0}p.npy'.format(prong)).flatten()

    if prong == 2:
        ax0 = plt.subplot(gs0[0])
        plt.hist(pt[labels==0], histtype='step', bins=30, range=(50, 2000))
        plt.yscale('log')
        plt.xlabel(r'$p_{T}$ [Gev]')
        plt.minorticks_on()
        plt.ylabel('Arb.')
        plt.title('QCD')

    ax1 = plt.subplot(gs0[prong - 1], sharey=ax0, sharex=ax0)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.hist(pt[labels==1], histtype='step', bins=30, range=(500, 2000))
    plt.yscale('log')
    plt.xlabel(r'$p_{T}$ [Gev]')
    plt.xlim(500, 2000)
    plt.xticks([500, 1000, 1500, 2000], [500, 1000, 1500, ''])
    plt.minorticks_on()
    plt.ylim(10, 1e5)
    plt.title('{0}-prong'.format(prong))
plt.savefig('reports/figures/TransverseMomentum.pdf', bbox_inches='tight')
