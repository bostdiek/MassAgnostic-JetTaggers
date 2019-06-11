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

data_dir = 'data/modelpredictions/'

for prong in [2, 3, 4]:
    with open(data_dir + 'ROCCurves_{0}p.p'.format(prong), 'rb') as f:
        ROC = pickle.load(f)

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)

    plt.figure(figsize=(8.5, 1.8))

    gs0 = gs.GridSpec(1, 3, width_ratios=[0.8, 0.8, 1.8], wspace=0.4)

    # ************************
    plt.subplot(gs0[0])
    plt.xlabel('Sig. Eff.')
    plt.ylabel('Back. Rej.')

    base, = plt.plot(ROC['BaseNeuralNetwork']['x_data'],
                     ROC['BaseNeuralNetwork']['y_data'],
                     color='C2'
                    )
    pca, = plt.plot(ROC['PCANeuralNetwork']['x_data'],
                    ROC['PCANeuralNetwork']['y_data'],
                    color='C3'
                   )
    planed, = plt.plot(ROC['PlanedNeuralNetwork']['x_data'],
                       ROC['PlanedNeuralNetwork']['y_data'],
                       color='C4'
                      )
    plt.plot(ROC['GradientBoostingClassifier']['x_data'],
             ROC['GradientBoostingClassifier']['y_data'],
             color='C2', ls =':'
            )
    plt.plot(ROC['PCAGBC']['x_data'],
             ROC['PCAGBC']['y_data'],
             color='C3', ls =':'
            )
    plt.plot(ROC['PlanedGBC']['x_data'],
             ROC['PlanedGBC']['y_data'],
             color='C4', ls =':'
            )
    if prong == 2:
        plt.legend([base, pca, planed],
                   ['Original', 'PCA', 'Planed'],
                   fontsize=10,
                   frameon=False
                  )
    plt.yscale('log')
    plt.ylim(1, 1e4)
    plt.xlim(0, 1)
    plt.minorticks_on()
    # ************************
    plt.subplot(gs0[1])
    plt.xlabel('Sig. Eff.')
    plt.ylabel('Bhat. Dist.')

    plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
             AllMets['BaseNeuralNetwork']['BhatD'],
             color='C2'
            )
    plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
             AllMets['PCANeuralNetwork']['BhatD'],
             color='C3'
            )
    plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
             AllMets['PlanedNeuralNetwork']['BhatD'],
             color='C4'
            )
    plt.plot(AllMets['GradientBoostingClassifier']['efficiencies'],
             AllMets['GradientBoostingClassifier']['BhatD'],
             color='C2', ls=':'
            )
    plt.plot(AllMets['PCAGBC']['efficiencies'],
             AllMets['PCAGBC']['BhatD'],
             color='C3', ls =':'
            )
    plt.plot(AllMets['PlanedGBC']['efficiencies'],
             AllMets['PlanedGBC']['BhatD'],
             color='C4', ls =':'
            )
    plt.plot([], [], color='k', label='NN')
    plt.plot([], [], color='k', ls=':', label='GBC')
    if prong == 2:
        plt.legend(fontsize=10, frameon=False)
    plt.ylim(0, 1.2)
    plt.xlim(0, 1)
    plt.minorticks_on()
    # ************************
    gs1 = gs.GridSpecFromSubplotSpec(1, 2, gs0[2], wspace=0.1)

    # ************************
    ax0 = plt.subplot(gs1[0])
    plt.xlabel('Sig. Eff.')
    plt.ylabel(r'$\sqrt{q_0}$')

    plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['BaseNeuralNetwork']['q0_0.01']),
             color='C2'
            )
    plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['PCANeuralNetwork']['q0_0.01']),
             color='C3'
            )
    plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['PlanedNeuralNetwork']['q0_0.01']),
             color='C4'
            )
    plt.plot(AllMets['GradientBoostingClassifier']['efficiencies'],
             np.sqrt(AllMets['GradientBoostingClassifier']['q0_0.01']),
             color='C2', ls =':'
            )
    plt.plot(AllMets['PCAGBC']['efficiencies'],
             np.sqrt(AllMets['PCAGBC']['q0_0.01']),
             color='C3', ls =':'
            )
    plt.plot(AllMets['PlanedGBC']['efficiencies'],
             np.sqrt(AllMets['PlanedGBC']['q0_0.01']),
             color='C4', ls =':'
            )
    plt.xlim(-0.05, 1.05)
    plt.minorticks_on()
    if prong == 2:
        plt.text(0.5, 3.3, 'Small Uncertainty', fontsize=10, ha='center', va='top')
    # ************************
    ax1 = plt.subplot(gs1[1], sharey=ax0)
    plt.xlabel('Sig. Eff.')
    plt.setp(ax1.get_yticklabels(), visible=False)

    plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['BaseNeuralNetwork']['q0_0.5']),
             color='C2'
            )
    plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['PCANeuralNetwork']['q0_0.5']),
             color='C3'
            )
    plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
             np.sqrt(AllMets['PlanedNeuralNetwork']['q0_0.5']),
             color='C4'
            )
    plt.plot(AllMets['GradientBoostingClassifier']['efficiencies'],
             np.sqrt(AllMets['GradientBoostingClassifier']['q0_0.5']),
             color='C2', ls =':'
            )
    plt.plot(AllMets['PCAGBC']['efficiencies'],
             np.sqrt(AllMets['PCAGBC']['q0_0.5']),
             color='C3', ls =':'
            )
    plt.plot(AllMets['PlanedGBC']['efficiencies'],
             np.sqrt(AllMets['PlanedGBC']['q0_0.5']),
             color='C4', ls =':'
            )
    plt.xlim(-0.05, 1.05)
    plt.minorticks_on()
    if prong == 2:
        plt.ylim(1, 3.5)
        plt.text(0.5, 3.3, 'Large Uncertainty', fontsize=10, ha='center', va='top')
    # ************************
    plt.suptitle('{0}-prong Signal'.format(prong), fontsize=14, y=1.02)
    plt.savefig('reports/figures/AugmentData_{0}p.pdf'.format(prong), bbox_inches='tight')
