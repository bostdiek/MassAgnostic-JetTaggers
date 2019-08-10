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
ADV_Colors9 = [
    '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
    '#e7298a', '#ce1256', '#980043', '#67001f'
    ]

# ***************************************************

# ROC Curves
plt.figure(figsize=(8.5, 9))
gs0 = gs.GridSpec(3, 3, wspace=0.1, hspace=0.35)
for prong in [2, 3, 4]:

    with open(data_dir + 'ROCCurves_{0}p.p'.format(prong), 'rb') as fin:
        ROCS = pickle.load(fin)

    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')
    # xr = np.linspace(1e-5, 1, 100)
    # plt.fill_between(xr, 1/xr, where=1/xr>0, color='grey')
    lam_plt_dict = {}
    lambdas_to_use = [1000, 100, 50, 20, 10, 5, 2, 1]
    for i, lam in enumerate(lambdas_to_use):
        lam_name = '{0:03d}'.format(lam)
        print(i, lam, lam_name)
        lamplt, = plt.plot(ROCS['AdversaryLambda_' + lam_name]['x_data'],
                           ROCS['AdversaryLambda_' + lam_name]['y_data'],
                           color=ADV_Colors9[i],
                           label=r'$\lambda=$' + '{0}'.format(lam)
                           )
        lam_plt_dict[lam] = lamplt
        print(prong, lam, ROCS['AdversaryLambda_' + lam_name]['auc'])
    if prong == 2:
        plt.ylabel('Background Rejection')
        # plt.legend(fontsize=10,
        #            frameon=True,
        #            loc='upper right',
        #            labelspacing=0.1
        #            )
        plt.legend(fontsize=10,
                   frameon=True,
                   loc='upper right',
                   labelspacing=0.15,
                   columnspacing=0.75,
                   handlelength=1.35,
                   ncol=2
                   )
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

# plt.savefig('reports/figures/AdversariesROCS.pdf',
#             bbox_inches='tight')
# plt.clf()
# plt.close()

# Bhattacharyya distance
# plt.figure(figsize=(8.5, 2.5))
# gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)
#
    ax2 = plt.subplot(gs0[prong + 1])
    plt.xlabel('Signal Efficiency')
    for i, lam in enumerate(lambdas_to_use):
        lam_name = '{0:03d}'.format(lam)
        # print(i, lam, lam_name)
        lamplt, = plt.plot(AllMets['AdversaryLambda_' + lam_name]['efficiencies'],
                           AllMets['AdversaryLambda_' + lam_name]['BhatD'],
                           color=ADV_Colors9[i],
                           label=r'$\lambda=$' + '{0}'.format(lam)
                           )
        lam_plt_dict[lam] = lamplt
    if prong == 2:
        plt.ylabel('Bhattacharyya distance')
    elif prong == 3:
        # plt.legend(fontsize=10,
        #            frameon=True,
        #            loc='upper right',
        #            labelspacing=0.1
        #            )
        plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        plt.setp(ax2.get_yticklabels(), visible=False)
    # plt.title('{0}-prong'.format(prong))
    plt.ylim(0, 1.)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0]
               )
    plt.xlim(-.05, 1.05)
    plt.grid()
    plt.minorticks_on()

# plt.savefig('reports/figures/AdversariesBhat.pdf',
#             bbox_inches='tight')
# plt.clf()
# plt.close()


# *********** Background rejection versus Distance ***********
# plt.figure(figsize=(8.5, 2.5))
# gs0 = gs.GridSpec(1, 3, wspace=0.1)

# axesdict = {2: plt.subplot(gs0[0]),
#             3: plt.subplot(gs0[1], sharey=ax0),
#             4: plt.subplot(gs0[2], sharey=ax0)
#             }
for prong in [2, 3, 4]:
    if prong == 2:
        ax = plt.subplot(gs0[prong + 4])
        plt.ylabel('Bhattacharyya Distance')
    else:
        ax1 = plt.subplot(gs0[prong + 4])  #,
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

    # interpolate the backfround rejections
    for i, lam in enumerate(lambdas_to_use):
        lam_name = '{0:03d}'.format(lam)
        print(i, lam, lam_name)
        backrej = interp1d(ROCS['AdversaryLambda_' + lam_name]['x_data'],
                           ROCS['AdversaryLambda_' + lam_name]['y_data'],
                           fill_value="extrapolate"
                           )
        dist = interp1d(AllMets['AdversaryLambda_' + lam_name]['efficiencies'],
                        AllMets['AdversaryLambda_' + lam_name]['BhatD'])
        lamplt, = plt.plot(backrej(AllMets['AdversaryLambda_' + lam_name]['efficiencies']),
                           dist(AllMets['AdversaryLambda_' + lam_name]['efficiencies']),
                           color=ADV_Colors9[i],
                           label=r'$\lambda=$' + '{0}'.format(lam)
                           )
        lam_plt_dict[lam] = lamplt
        plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color=ADV_Colors9[i], zorder=10)
        plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color=ADV_Colors9[i], zorder=10)
        plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color=ADV_Colors9[i], zorder=10)

    # plt.title('{0}-prong'.format(prong))
    if prong == 2:
        plt.ylabel('Bhattacharyya distance')
        # plt.legend(fontsize=10,
        #            frameon=False,
        #            loc='upper right',
        #            labelspacing=0.15,
        #            columnspacing=0.75,
        #            handlelength=1.35,
        #            ncol=2
        #            )
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
    elif prong == 3:
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
plt.savefig('reports/figures/AdversariesAppendix.pdf',
            bbox_inches='tight')
