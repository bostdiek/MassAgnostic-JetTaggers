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

Methods = [  # 'BaseNeuralNetwork', 'GradientBoostingClassifier', 'TauSubjettiness',
           'uBoost', 'AdversaryLambda_050',
           'PCANeuralNetwork', 'PlanedNeuralNetwork', 'TauDDT'
           ]
LS = [  # '-', '--', ':',
      '--', '-',
      '-', '-', ':'
      ]
MyColors = [  # 'C0', 'C0', 'hotpink',
            'blue', 'C4',
            'C3', 'C2', 'purple'
            ]
LegendLabels = [  # 'NN', 'BDT', r'$\tau_{N}/\tau_{N-1}$',
                'uBoost', 'Adv',
                'PCA', 'Planed', r'$\tau_{21}^{\rm{DDT}}$'
                ]
plt.figure(figsize=(8.5, 5.7))
gs0 = gs.GridSpec(2, 3, wspace=0.1, hspace=0.35)

# axesdict = {2: plt.subplot(gs0[0]),
#             3: plt.subplot(gs0[1], sharey=ax0),
#             4: plt.subplot(gs0[2], sharey=ax0)
#             }
for prong in [2, 3, 4]:
    if prong == 2:
        ax = plt.subplot(gs0[prong + 1])
        plt.ylabel('Bhattacharyya Distance')
    else:
        ax1 = plt.subplot(gs0[prong + 1])  #,
                          # sharey=ax)
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

    for method, color, ls in zip(Methods, MyColors, LS):
        if (prong != 2) and (method == 'TauDDT'):
            continue
        # interpolate the backfround rejections
        backrej = interp1d(ROCS[method]['x_data'],
                           ROCS[method]['y_data'],
                           fill_value="extrapolate"
                           )
        dist = interp1d(AllMets[method]['efficiencies'],
                        AllMets[method]['BhatD'])
        plt.plot(backrej(AllMets[method]['efficiencies']),
                 dist(AllMets[method]['efficiencies']),
                 # lw=1,
                 color=color,
                 ls=ls
                 )
        plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color=color, zorder=10)
        plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color=color, zorder=10)
        plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color=color, zorder=10)

    # plt.title('{0}-prong'.format(prong))
    if prong == 2:
        for color, label, ls in zip(MyColors, LegendLabels, LS):
            plt.plot([], [], color=color, label=label, ls=ls)
        # plt.legend(frameon=True,
        #            fontsize=10,
        #            loc='upper center',
        #            ncol=2,
        #            columnspacing=1,
        #            labelspacing=0.15)
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
    if prong == 3:
        plt.xlim(2e3, 0.8)
    if prong == 4:
        eff50 = plt.scatter([], [], marker='*', s=36, color='k', zorder=10)
        eff25 = plt.scatter([], [], marker='s', s=25, color='k', zorder=10)
        eff75 = plt.scatter([], [], marker='o', s=25, color='k', zorder=10)
        plt.legend([eff75, eff50, eff25],
                   [r'$\varepsilon_{S} = 0.75$',
                    r'$\varepsilon_{S} = 0.50$',
                    r'$\varepsilon_{S} = 0.25$'
                    ],
                   frameon=True,
                   fontsize=10,
                   loc='upper right'
                   )
    plt.grid()
# plt.savefig('reports/figures/AllMethodsBackgroundRejectionVersusDistance.pdf', bbox_inches='tight')
# plt.close()

# ************************************
# Bhattacharyya distance
# ************************************
lam_plt_dict = {}
lambdas_to_use = [50]
Uboost_color = 'blue'
# plt.figure(figsize=(8.5, 2.5))
# gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)
#
    if prong == 2:
        ax = plt.subplot(gs0[prong - 2])
        plt.ylabel('Bhattacharyya Distance')
    else:
        ax1 = plt.subplot(gs0[prong - 2])
        plt.setp(ax1.get_yticklabels(), visible=False)
    plt.xlabel('Signal Efficiency')
    for method, color, ls in zip(Methods, MyColors, LS):
        if (prong != 2) and (method == 'TauDDT'):
            continue
        # print(i, lam, lam_name)
        lamplt, = plt.plot(AllMets[method]['efficiencies'],
                           AllMets[method]['BhatD'],
                           color=color,
                           ls=ls,
                           # label=r'$\lambda=$' + '{0}'.format(lam)
                           )
    plt.title('{0}-prong'.format(prong))
    plt.ylim(0, 1.2)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()
    if prong == 2:
        for color, label, ls in zip(MyColors, LegendLabels, LS):
            plt.plot([], [], color=color, label=label, ls=ls)
        plt.legend(frameon=True,
                   fontsize=10,
                   loc='upper center',
                   ncol=2,
                   columnspacing=1,
                   labelspacing=0.15)


plt.savefig('reports/figures/AllMethodsBhat_Both.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()

# ************************************
# ROCS
# ************************************
lam_plt_dict = {}
lambdas_to_use = [50]
Uboost_color = 'blue'
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'ROCCurves_{0}p.p'.format(prong), 'rb') as fin:
        ROCS = pickle.load(fin)
#
    if prong == 2:
        ax = plt.subplot(gs0[prong - 2])
        plt.ylabel('Background Rejection')
    else:
        ax1 = plt.subplot(gs0[prong - 2])
        plt.setp(ax1.get_yticklabels(), visible=False)
    plt.xlabel('Signal Efficiency')
    for method, color, ls in zip(Methods, MyColors, LS):
        if (prong != 2) and (method == 'TauDDT'):
            continue
        # print(i, lam, lam_name)
        lamplt, = plt.plot(ROCS[method]['x_data'],
                           ROCS[method]['y_data'],
                           color=color,
                           ls=ls,
                           # label=r'$\lambda=$' + '{0}'.format(lam)
                           )
    plt.title('{0}-prong'.format(prong))
    plt.yscale('log')
    plt.ylim(1, 1e4)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()
    if prong == 2:
        for color, label, ls in zip(MyColors, LegendLabels, LS):
            plt.plot([], [], color=color, label=label, ls=ls)
        plt.legend(frameon=True,
                   fontsize=10,
                   loc='upper center',
                   ncol=2,
                   columnspacing=1,
                   labelspacing=0.15)


plt.savefig('reports/figures/AllMethodsROCS.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()
