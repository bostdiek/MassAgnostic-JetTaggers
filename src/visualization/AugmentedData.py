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

data_dir = 'data/modelpredictions/'

# ROC Curves
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'ROCCurves_{0}p.p'.format(prong), 'rb') as fin:
        ROCS = pickle.load(fin)

    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')
    # xr = np.linspace(1e-5, 1, 100)
    # plt.fill_between(xr, 1/xr, where=1/xr>0, color='grey')

    base, = plt.plot(ROCS['BaseNeuralNetwork']['x_data'],
                     ROCS['BaseNeuralNetwork']['y_data'],
                     color='C2'
                     )
    pca, = plt.plot(ROCS['PCANeuralNetwork']['x_data'],
                    ROCS['PCANeuralNetwork']['y_data'],
                    color='C3'
                    )
    planed, = plt.plot(ROCS['PlanedNeuralNetwork']['x_data'],
                       ROCS['PlanedNeuralNetwork']['y_data'],
                       color='C4'
                       )
    plt.plot(ROCS['GradientBoostingClassifier']['x_data'],
             ROCS['GradientBoostingClassifier']['y_data'],
             color='C2', ls=':'
             )
    plt.plot(ROCS['PCAGBC']['x_data'],
             ROCS['PCAGBC']['y_data'],
             color='C3', ls=':'
             )
    plt.plot(ROCS['PlanedGBC']['x_data'],
             ROCS['PlanedGBC']['y_data'],
             color='C4', ls=':'
             )
    nn, = plt.plot([], [], color='k', label='NN')
    gbc, = plt.plot([], [], color='k', ls=':', label='GBC')
    if prong == 2:
        plt.ylabel('Background Rejection')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([base, pca, planed],
                   ['Original', 'PCA', 'Planed'],
                   fontsize=10,
                   frameon=True
                   )
    elif prong == 3:
        plt.legend([nn, gbc],
                   ['NN', 'GBC'],
                   fontsize=10,
                   frameon=True,
                   )
        plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        plt.setp(ax2.get_yticklabels(), visible=False)
    plt.title('{0}-prong'.format(prong))
    plt.ylim(1, 1e4)
    plt.yscale('log')
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()

plt.savefig('reports/figures/AugmentDataROCS.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()

# Bhattacharyya distance
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)

    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')

    base, = plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
                     AllMets['BaseNeuralNetwork']['BhatD'],
                     color='C2'
                     )
    pca, = plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
                    AllMets['PCANeuralNetwork']['BhatD'],
                    color='C3'
                    )
    planed, = plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
                       AllMets['PlanedNeuralNetwork']['BhatD'],
                       color='C4'
                       )
    plt.plot(AllMets['GradientBoostingClassifier']['efficiencies'],
             AllMets['GradientBoostingClassifier']['BhatD'],
             color='C2', ls=':'
             )
    plt.plot(AllMets['PCAGBC']['efficiencies'],
             AllMets['PCAGBC']['BhatD'],
             color='C3', ls=':'
             )
    plt.plot(AllMets['PlanedGBC']['efficiencies'],
             AllMets['PlanedGBC']['BhatD'],
             color='C4', ls=':'
             )
    nn, = plt.plot([], [], color='k', label='NN')
    gbc, = plt.plot([], [], color='k', ls=':', label='GBC')
    if prong == 2:
        plt.ylabel('Bhattacharyya distance')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([base, pca, planed],
                   ['Original', 'PCA', 'Planed'],
                   fontsize=10,
                   frameon=True
                   )
    elif prong == 3:
        plt.legend([nn, gbc],
                   ['NN', 'GBC'],
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

plt.savefig('reports/figures/AugmentDataBhat.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()

# *********** q0 ***********
plt.figure(figsize=(8.5, 7.5))
gs0 = gs.GridSpec(3, 3, wspace=0.05, hspace=0.05)
for prong in [2, 3, 4]:
    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)

    ax_dict = {}
    titles = ['1% uncertainty', '10% uncertainty', '50% uncertainty']
    for i, metric in enumerate(['q0_0.01', 'q0_0.1', 'q0_0.5']):
        if i == 0:
            ax0 = plt.subplot(gs0[(prong - 2) * 3])
        else:
            ax = plt.subplot(gs0[(prong - 2) * 3 + i], sharey=ax0)
        print(np.sqrt(AllMets['BaseNeuralNetwork'][metric]))
        xspace = np.linspace(0.05, 1, 50)
        effarray = np.array(AllMets['BaseNeuralNetwork']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['BaseNeuralNetwork'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
              bounds_error=False
                            )
        plt.plot(xspace,
                 q0interp(xspace),
                 color='C2'
                 )
        effarray = np.array(AllMets['PCANeuralNetwork']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['PCANeuralNetwork'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
              bounds_error=False
                            )
        plt.plot(xspace,
                 q0interp(xspace),
                 color='C3'
                 )

        effarray = np.array(AllMets['PlanedNeuralNetwork']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['PlanedNeuralNetwork'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
              bounds_error=False
                            )
        plt.plot(xspace,
                 q0interp(xspace),
                 color='C4'
                 )

        effarray = np.array(AllMets['GradientBoostingClassifier']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['GradientBoostingClassifier'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                            fill_value=[0],
                            bounds_error=False
                            )
        plt.plot(xspace, q0interp(xspace),
                 color='C2', ls=':'
                 )

        effarray = np.array(AllMets['PCAGBC']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['PCAGBC'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
              bounds_error=False
                            )
        plt.plot(xspace, q0interp(xspace),
                 color='C3', ls=':'
                 )

        effarray = np.array(AllMets['PlanedGBC']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['PlanedGBC'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
              bounds_error=False
                            )
        plt.plot(xspace, q0interp(xspace),
                 color='C4', ls=':'
                 )
        if i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel(r'$\sqrt{q_0}$')
        if prong == 4:
            plt.xlabel('Signal Efficiency')
        else:
            if i == 0:
                plt.setp(ax0.get_xticklabels(), visible=False)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
        if prong == 2:
            plt.title(titles[i])
        if i == 2 and prong == 2:
            plt.ylim(0.5, 3)
        if i == 0 and prong == 2:
            plt.text(0.5, 0.65, '2-prong signal', ha='center', va='bottom')
        if i == 2 and prong == 3:
            plt.ylim(1, 7.2)
        if i == 0 and prong == 3:
            plt.text(0.5, 1.4, '3-prong signal', ha='center', va='bottom')
        if i == 2 and prong == 4:
            plt.ylim(1, 16.2)
        if i == 0 and prong == 4:
            plt.text(0.5, 1.8, '4-prong signal', ha='center', va='bottom')
        if i == 2 and prong == 2:
            plt.legend([base, pca, planed],
                       ['Original', 'PCA', 'Planed'],
                       fontsize=10,
                       frameon=False
                       )
        if i == 1 and prong == 2:
            plt.legend([nn, gbc],
                       ['NN', 'GBC'],
                       fontsize=10,
                       frameon=False)
        plt.xlim(-0.05, 1.05)
        plt.minorticks_on()
plt.savefig('reports/figures/AugmentDataSig.pdf',
            bbox_inches='tight')
