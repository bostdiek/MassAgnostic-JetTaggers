import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from pathlib import Path
import pickle
from scipy.interpolate import interp1d

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 12,
                     'axes.unicode_minus': False,
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
project_dir = Path(__file__).resolve().parents[2]
p_name = project_dir.joinpath('data/modelpredictions/')
data_dir = str(p_name.resolve())
data_dir = 'data/modelpredictions/'

for prong in [2, 3, 4]:

    with open(data_dir + '/Histograms_{0}p.p'.format(prong), 'rb') as f:
        Hist = pickle.load(f)


    plt.figure(figsize=(8, 10))
    gs0 = gs.GridSpec(4, 3, wspace=0.15, hspace=0.15)
    plot_keys = [0.95, 0.9, 0.8, 0.70, 0.60, 0.5]
    mycolors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C8']
    # ************** TauN ************************
    ax0 = plt.subplot(gs0[0])
    hists = Hist['TauSubjettiness']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.text(450 / 2, 7e3, r'$\tau_{N}/\tau_{N-1}$', fontsize=12, ha='center', va='top')
    plt.ylabel('Arb.')

    if prong == 2:
        plt.setp(ax0.get_xticklabels(), visible=False)
    else:
        plt.xlabel(r'$m_J$ [GeV]')

    # ************** GradientBoostingClassifier ************************
    ax1 = plt.subplot(gs0[1], sharey=ax0, sharex=ax0)
    hists = Hist['GradientBoostingClassifier']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'BDT', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ************** Neural Network ************************
    ax1 = plt.subplot(gs0[2], sharey=ax0, sharex=ax0)
    hists = Hist['BaseNeuralNetwork']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'NN', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ************** TauDDT ************************
    if prong == 2:
        ax0 = plt.subplot(gs0[3], sharey=ax0, sharex=ax0)
        hists = Hist['TauDDT']
        plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
        for i, eff in enumerate(plot_keys):
            plt.hist(hists[eff][1],
                     range=(50, 400),
                     histtype='step',
                     bins=35,
                     color=mycolors[i]
                     )
        plt.hist(hists[1][0], range=(50, 400),
                 bins=35,
                 color='grey',
                 alpha=0.4,
                 weights=0.2 * np.ones_like(hists[1][0]))
        plt.xlim(50, 400)
        plt.yscale('log')
        plt.minorticks_on()
        plt.text(450 / 2, 7e3, r'$\tau_{21}^{\rm{DDT}}$', fontsize=12, ha='center', va='top')
        plt.xlabel(r'$m_J$ [GeV]')
        plt.ylabel('Arb.')
    # ************** uBoost ************************
    ax1 = plt.subplot(gs0[4], sharey=ax0, sharex=ax0)
    hists = Hist['uBoost']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'uBoost', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    if prong == 2:
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel('Arb.')
    # ************** AdversaryLambda_050 ************************
    ax1 = plt.subplot(gs0[5], sharey=ax0, sharex=ax0)
    hists = Hist['AdversaryLambda_050']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'Adv', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ************** Planed BDT ************************
    ax1 = plt.subplot(gs0[7], sharey=ax0, sharex=ax0)
    hists = Hist['PlanedGBC']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'Planed BDT', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    # plt.setp(ax1.get_yticklabels(), visible=False)
    plt.ylabel('Arb.')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ************** Planed NN ************************
    ax1 = plt.subplot(gs0[8], sharey=ax0, sharex=ax0)
    hists = Hist['PlanedNeuralNetwork']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'Planed NN', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ************** Legend ************************
    ax1 = plt.subplot(gs0[9], sharey=ax0, sharex=ax0)
    for eff, color in zip(plot_keys, mycolors):
        plt.plot([], [], color=color, label=r'$\varepsilon_S=$ {0:.2f}'.format(eff))
    plt.legend(loc='center left', frameon=False)
    plt.axis('off')
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # ************** PCA BDT ************************
    ax1 = plt.subplot(gs0[10], sharey=ax0, sharex=ax0)
    hists = Hist['PCAGBC']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'PCA BDT', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.ylabel('Arb.')
    # plt.setp(ax1.get_yticklabels(), visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel(r'$m_J$ [GeV]')

    # ************** Planed NN ************************
    ax1 = plt.subplot(gs0[11], sharey=ax0, sharex=ax0)
    hists = Hist['PCANeuralNetwork']
    plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
    for i, eff in enumerate(plot_keys):
        plt.hist(hists[eff][1],
                 range=(50, 400),
                 histtype='step',
                 bins=35,
                 color=mycolors[i]
                 )
    plt.text(450 / 2, 7e3, 'PCA NN', fontsize=12, ha='center', va='top')
    plt.hist(hists[1][0], range=(50, 400),
             bins=35,
             color='grey',
             alpha=0.4,
             weights=0.2 * np.ones_like(hists[1][0]))
    plt.xlim(50, 400)
    plt.yscale('log')
    plt.minorticks_on()
    plt.setp(ax1.get_yticklabels(), visible=False)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlabel(r'$m_J$ [GeV]')
    # ************** final parts ************************
    plt.xlim(50, 400)
    plt.xticks([100, 200, 300, 400])
    plt.ylim(1e1, 1e4)
    plt.yticks([10, 100, 1000, 10000])
    plt.minorticks_on()
    plt.suptitle('{0}-prong signal'.format(prong), y=0.91, fontsize=14)
    plt.savefig('reports/figures/AllHistos_{0}.pdf'.format(prong),
                bbox_inches='tight')
