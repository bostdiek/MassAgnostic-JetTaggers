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

nn, = plt.plot([], [], color='k', label='NN')
gbc, = plt.plot([], [], color='k', ls='--', label='BDT')
singlevar, = plt.plot([], [], color='k', ls=':', label='Single Variable')
plt.clf()
plt.close()

# Bhattacharyya distance
plt.figure(figsize=(8.5, 5.7))
gs0 = gs.GridSpec(2, 3, wspace=0.1, hspace=0.35)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)

    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')

    base, = plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
                     AllMets['BaseNeuralNetwork']['BhatD'],
                     color='C0'
                     )
    pca, = plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
                    AllMets['PCANeuralNetwork']['BhatD'],
                    color='C3'
                    )
    planed, = plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
                       AllMets['PlanedNeuralNetwork']['BhatD'],
                       color='C2'
                       )
    taunn, = plt.plot(AllMets['TauSubjettiness']['efficiencies'],
                      AllMets['TauSubjettiness']['BhatD'],
                      color='hotpink',
                      ls=':',
                      zorder=10
                      )
    if prong == 2:
        taunnddt, = plt.plot(AllMets['TauDDT']['efficiencies'],
                             AllMets['TauDDT']['BhatD'],
                             color='purple',
                             ls=':',
                             zorder=10
                             )
    plt.plot(AllMets['GradientBoostingClassifier']['efficiencies'],
             AllMets['GradientBoostingClassifier']['BhatD'],
             color='C0', ls='--'
             )
    plt.plot(AllMets['PCAGBC']['efficiencies'],
             AllMets['PCAGBC']['BhatD'],
             color='C3', ls='--'
             )
    plt.plot(AllMets['PlanedGBC']['efficiencies'],
             AllMets['PlanedGBC']['BhatD'],
             color='C2', ls='--'
             )
    if prong == 2:
        plt.ylabel('Bhattacharyya Distance')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([base, pca, planed, taunn, taunnddt],
                   ['Original', 'PCA', 'Planed', r'$\tau_N / \tau_{N-1}$',
                    r'$\tau_{21}^{\rm{DDT}}$'],
                   fontsize=10,
                   frameon=True,
                   labelspacing=0.15
                   )
    elif prong == 3:
        plt.legend([nn, gbc],  # , singlevar],
                   ['NN', 'BDT'],  # , 'Single Variable'],
                   fontsize=10,
                   frameon=True)
        plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        plt.setp(ax2.get_yticklabels(), visible=False)
    plt.title('{0}-prong'.format(prong))
    plt.ylim(0, 1.2)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()

# ********** SECOND ROW ***************************
for prong in [2, 3, 4]:
    if prong == 2:
        ax = plt.subplot(gs0[prong + 1])
        plt.ylabel('Bhattacharyya Distance')
    else:
        ax1 = plt.subplot(gs0[prong + 1])
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.xlim(2e4, 0.8)
    plt.xlabel('Background Rejection')
    plt.xscale('log')
    # plt.xlim(2e4, 0.8)
    plt.xticks([1e4, 1e3, 1e2, 1e1, 1e0])
    plt.ylim(-0.05, 1.0)
    plt.minorticks_on()
    # Load the data
    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)
    with open(data_dir + 'ROCCurves_{0}p.p'.format(prong), 'rb') as fin:
        ROCS = pickle.load(fin)

    # interpolate the backfround rejections
    backrej = interp1d(ROCS['BaseNeuralNetwork']['x_data'],
                       ROCS['BaseNeuralNetwork']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['BaseNeuralNetwork']['efficiencies'],
                    AllMets['BaseNeuralNetwork']['BhatD'])
    base, = plt.plot(backrej(AllMets['BaseNeuralNetwork']['efficiencies']),
                     dist(AllMets['BaseNeuralNetwork']['efficiencies']),
                     color='C0'
                     )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C0', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C0', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C0', zorder=10)

    backrej = interp1d(ROCS['PCANeuralNetwork']['x_data'],
                       ROCS['PCANeuralNetwork']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['PCANeuralNetwork']['efficiencies'],
                    AllMets['PCANeuralNetwork']['BhatD'])
    pca, = plt.plot(backrej(AllMets['PCANeuralNetwork']['efficiencies']),
                    dist(AllMets['PCANeuralNetwork']['efficiencies']),
                    color='C3'
                    )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C3', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C3', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C3', zorder=10)

    backrej = interp1d(ROCS['PlanedNeuralNetwork']['x_data'],
                       ROCS['PlanedNeuralNetwork']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['PlanedNeuralNetwork']['efficiencies'],
                    AllMets['PlanedNeuralNetwork']['BhatD'])
    planed, = plt.plot(backrej(AllMets['PlanedNeuralNetwork']['efficiencies']),
                       dist(AllMets['PlanedNeuralNetwork']['efficiencies']),
                       color='C2'
                       )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C2', zorder=10)

    backrej = interp1d(ROCS['GradientBoostingClassifier']['x_data'],
                       ROCS['GradientBoostingClassifier']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['GradientBoostingClassifier']['efficiencies'],
                    AllMets['GradientBoostingClassifier']['BhatD'])
    plt.plot(backrej(AllMets['GradientBoostingClassifier']['efficiencies']),
             dist(AllMets['GradientBoostingClassifier']['efficiencies']),
             color='C0',
             ls='--'
             )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C0', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C0', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C0', zorder=10)

    backrej = interp1d(ROCS['PCAGBC']['x_data'],
                       ROCS['PCAGBC']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['PCAGBC']['efficiencies'],
                    AllMets['PCAGBC']['BhatD'])
    plt.plot(backrej(AllMets['PCAGBC']['efficiencies']),
             dist(AllMets['PCAGBC']['efficiencies']),
             color='C3',
             ls='--'
             )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C3', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C3', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C3', zorder=10)

    backrej = interp1d(ROCS['PlanedGBC']['x_data'],
                       ROCS['PlanedGBC']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['PlanedGBC']['efficiencies'],
                    AllMets['PlanedGBC']['BhatD'])
    plt.plot(backrej(AllMets['PlanedGBC']['efficiencies']),
             dist(AllMets['PlanedGBC']['efficiencies']),
             color='C2',
             ls='--'
             )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C2', zorder=10)

    backrej = interp1d(ROCS['PlanedGBC']['x_data'],
                       ROCS['PlanedGBC']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['PlanedGBC']['efficiencies'],
                    AllMets['PlanedGBC']['BhatD'])
    plt.plot(backrej(AllMets['PlanedGBC']['efficiencies']),
             dist(AllMets['PlanedGBC']['efficiencies']),
             color='C2',
             ls='--'
             )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='C2', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='C2', zorder=10)

    nn, = plt.plot([], [], color='k', label='NN')
    gbc, = plt.plot([], [], color='k', ls='--', label='BDT')

    backrej = interp1d(ROCS['TauSubjettiness']['x_data'],
                       ROCS['TauSubjettiness']['y_data'],
                       fill_value="extrapolate"
                       )
    dist = interp1d(AllMets['TauSubjettiness']['efficiencies'],
                    AllMets['TauSubjettiness']['BhatD'])
    taunn, = plt.plot(backrej(AllMets['TauSubjettiness']['efficiencies']),
                      dist(AllMets['TauSubjettiness']['efficiencies']),
                      color='hotpink',
                      ls=':'
                      )
    plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='hotpink', zorder=10)
    plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='hotpink', zorder=10)
    plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='hotpink', zorder=10)

    if prong == 2:
        backrej = interp1d(ROCS['TauDDT']['x_data'],
                           ROCS['TauDDT']['y_data'],
                           fill_value="extrapolate"
                           )
        dist = interp1d(AllMets['TauDDT']['efficiencies'],
                        AllMets['TauDDT']['BhatD'])
        taunnddt, = plt.plot(backrej(AllMets['TauDDT']['efficiencies']),
                             dist(AllMets['TauDDT']['efficiencies']),
                             color='purple',
                             ls=':'
                             )
        plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='purple', zorder=10)
        plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='purple', zorder=10)
        plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='purple', zorder=10)

    # plt.title('{0}-prong'.format(prong))
    if prong == 2:
        plt.ylabel('Bhattacharyya Distance')
        ax.annotate("Better",
                    ha='center',
                    fontsize=10,
                    xy=(7.5e2, 0.06),
                    xytext=(7.5e2, 0.42),
                    rotation=90,
                    arrowprops=dict(arrowstyle="->")
                    )
        ax.annotate("Better",
                    va='center',
                    fontsize=10,
                    xy=(7e2, 0.05),
                    xytext=(1e2, 0.05),
                    arrowprops=dict(arrowstyle="->")
                    )
        plt.xlim(2e3, 0.8)
    if prong == 4:
        eff50 = plt.scatter([], [], marker='*', s=25, color='k', zorder=10)
        eff25 = plt.scatter([], [], marker='s', s=25, color='k', zorder=10)
        eff75 = plt.scatter([], [], marker='o', s=25, color='k', zorder=10)
        plt.legend([eff75, eff50, eff25],
                   [r'$\epsilon_{S} = 0.75$',
                    r'$\epsilon_{S} = 0.50$',
                    r'$\epsilon_{S} = 0.25$'
                    ],
                   frameon=True,
                   fontsize=10,
                   loc='upper right'
                   )
    plt.grid()

plt.savefig('reports/figures/CombinedBhatDistAugmentData.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()
