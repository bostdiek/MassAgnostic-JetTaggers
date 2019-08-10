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

# Bhattacharyya distance
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)

    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')
    print(AllMets['BaseNeuralNetwork'])
    base, = plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
                     AllMets['BaseNeuralNetwork']['BhatD'],
                     color='C0'
                     )

    plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
             AllMets['BaseNeuralNetwork']['JSD'],
             color='C0', ls='--'
             )
    pca, = plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
                    AllMets['PCANeuralNetwork']['BhatD'],
                    color='C3'
                    )
    plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
             AllMets['PCANeuralNetwork']['JSD'],
             color='C3',
             ls='--'
             )
    planed, = plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
                       AllMets['PlanedNeuralNetwork']['BhatD'],
                       color='C2'
                       )
    plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
             AllMets['PlanedNeuralNetwork']['JSD'],
             color='C2',
             ls='--'
             )
    uboost, = plt.plot(AllMets['uBoost']['efficiencies'],
                       AllMets['uBoost']['BhatD'],
                       color='blue',
                       )
    plt.plot(AllMets['uBoost']['efficiencies'],
             AllMets['uBoost']['JSD'],
             color='blue',
             ls='--'
             )
    adversary, = plt.plot(AllMets['AdversaryLambda_050']['efficiencies'],
                          AllMets['AdversaryLambda_050']['BhatD'],
                          color='C4',
                          )
    plt.plot(AllMets['AdversaryLambda_050']['efficiencies'],
             AllMets['AdversaryLambda_050']['JSD'],
             color='C4',
             ls='--'
             )

    if prong == 2:
        plt.ylabel('Histogram Distance')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([base, pca, planed, uboost, adversary],
                   ['Original', 'PCA', 'Planed', 'uBoost',
                    'Adv'],
                   fontsize=10,
                   frameon=True,
                   labelspacing=0.15
                   )
    elif prong == 3:
        BD, = plt.plot([], [], color='gray')
        JS, = plt.plot([], [], color='gray', ls='--')
        plt.legend([BD, JS],  # , singlevar],
                   ['Bhattacharyya', 'Jensen-Shannon'],  # , 'Single Variable'],
                   fontsize=10,
                   frameon=True)
        plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        plt.setp(ax2.get_yticklabels(), visible=False)
    plt.title('{0}-prong'.format(prong))
    plt.ylim(0, 1.0)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()

plt.savefig('reports/figures/DistanceCompare.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()
