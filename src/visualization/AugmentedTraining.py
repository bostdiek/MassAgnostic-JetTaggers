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
# ['#f1eef6','#d7b5d8','#df65b0','#dd1c77','#980043']
ADV_Colors9 = [  # '#f7f4f9', , ,
               '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0',
               '#e7298a', '#ce1256', '#980043', '#67001f']
Uboost_color = 'blue'  # '#4dac26'

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
    lam_plt_dict = {}
    lambdas_to_use = [1000, 100, 50, 20, 10, 5, 2, 1]
    for i, lam in enumerate(lambdas_to_use):
        lam_name = '{0:03d}'.format(lam)
        # print(i, lam, lam_name)
        lamplt, = plt.plot(ROCS['AdversaryLambda_' + lam_name]['x_data'],
                           ROCS['AdversaryLambda_' + lam_name]['y_data'],
                           color=ADV_Colors9[i],
                           # label=r'$\lambda=$' + '{0}'.format(lam)
                           )
        lam_plt_dict[lam] = lamplt
        print(prong, lam, ROCS['AdversaryLambda_' + lam_name]['auc'])
    ubst, = plt.plot(ROCS['uBoost']['x_data'],
                     ROCS['uBoost']['y_data'],
                     color=Uboost_color,
                     ls='--', lw=1.5
                     )
    if prong == 4:
        plt.legend([lam_plt_dict[x] for x in lambdas_to_use[::-1]],
                   [str(x) for x in lambdas_to_use[::-1]],
                   fontsize=10,
                   frameon=True,
                   ncol=2,
                   loc='lower left',
                   columnspacing=1,
                   handlelength=1.2,
                   title=r'Adversary $\lambda$',
                   title_fontsize=10
                   )
    if prong == 2:
        plt.ylabel('Background Rejection')
        plt.legend([lam_plt_dict[5], ubst],
                   ['Adv. NN.', 'uBoost'],
                   fontsize=10,
                   frameon=True,
                   loc='upper right'
                   )
    # elif prong == 3:
    #     plt.legend([nn, gbc],
    #                ['NN', 'GBC'],
    #                fontsize=10,
    #                frameon=True,
    #                )
        # plt.setp(ax2.get_yticklabels(), visible=False)
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

plt.savefig('reports/figures/AugmentTrainingROCS.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()

# Bhattacharyya distance
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1,)
for prong in [2, 3, 4]:

    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)
#
    ax2 = plt.subplot(gs0[prong - 2])
    plt.xlabel('Signal Efficiency')
    for i, lam in enumerate(lambdas_to_use):
        lam_name = '{0:03d}'.format(lam)
        # print(i, lam, lam_name)
        lamplt, = plt.plot(AllMets['AdversaryLambda_' + lam_name]['efficiencies'],
                           AllMets['AdversaryLambda_' + lam_name]['BhatD'],
                           color=ADV_Colors9[i],
                           # label=r'$\lambda=$' + '{0}'.format(lam)
                           )
        lam_plt_dict[lam] = lamplt
    ubst, = plt.plot(AllMets['uBoost']['efficiencies'],
                     AllMets['uBoost']['BhatD'],
                     color=Uboost_color,
                     ls='--', lw=1.5
                     )
    if prong == 2:
        plt.ylabel('Bhattacharyya distance')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([lam_plt_dict[5], ubst],
                   ['Adv. NN.', 'uBoost'],
                   fontsize=10,
                   frameon=True,
                   loc='upper right'
                   )
    elif prong == 3:
        plt.legend([lam_plt_dict[x] for x in lambdas_to_use[::-1]],
                   [str(x) for x in lambdas_to_use[::-1]],
                   fontsize=10,
                   frameon=True,
                   ncol=2,
                   loc='upper right',
                   columnspacing=1,
                   handlelength=1.2,
                   title=r'Adversary $\lambda$',
                   title_fontsize=10
                   )
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

plt.savefig('reports/figures/AugmentTrainingBhat.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()
#
# # *********** q0 ***********
plt.figure(figsize=(8.5, 7.5))
gs0 = gs.GridSpec(3, 3, wspace=0.05, hspace=0.05)
for prong in [2, 3, 4]:
    with open(data_dir + 'Metrics_{0}p.p'.format(prong), 'rb') as fin:
        AllMets = pickle.load(fin)
#
    ax_dict = {}
    titles = ['1% uncertainty', '10% uncertainty', '50% uncertainty']
    for i, metric in enumerate(['q0_0.01', 'q0_0.1', 'q0_0.5']):
        if i == 0:
            ax0 = plt.subplot(gs0[(prong - 2) * 3])
        else:
            ax = plt.subplot(gs0[(prong - 2) * 3 + i], sharey=ax0)
        print(np.sqrt(AllMets['AdversaryLambda_' + lam_name][metric]))
        xspace = np.linspace(0.05, 1, 50)
        for j, lam in enumerate(lambdas_to_use):
            lam_name = '{0:03d}'.format(lam)
            effarray = np.array(AllMets['AdversaryLambda_' + lam_name]['efficiencies'])
            qoarray = np.array(np.sqrt(AllMets['AdversaryLambda_' + lam_name][metric]))
            q0interp = interp1d(effarray[~np.isinf(qoarray)],
                                qoarray[~np.isinf(qoarray)],
                            fill_value=[0],
                            bounds_error=False
                            )
            plt.plot(xspace, q0interp(xspace), color=ADV_Colors9[j])

        effarray = np.array(AllMets['uBoost']['efficiencies'])
        qoarray = np.array(np.sqrt(AllMets['uBoost'][metric]))
        q0interp = interp1d(effarray[~np.isinf(qoarray)],
                            qoarray[~np.isinf(qoarray)],
                        fill_value=[0],
                        bounds_error=False
                        )
        plt.plot(xspace, q0interp(xspace), color=Uboost_color, ls='--', lw=1.5)

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
            plt.text(0.6, 1.8, '4-prong signal', ha='center', va='bottom')
        if i == 2 and prong == 3:
            plt.legend([lam_plt_dict[x] for x in lambdas_to_use[::-1]],
                       [str(x) for x in lambdas_to_use[::-1]],
                       fontsize=10,
                       frameon=False,
                       ncol=2,
                       loc='upper right',
                       columnspacing=1,
                       handlelength=1.2,
                       title=r'Adversary $\lambda$',
                       title_fontsize=10
                       )
        if i == 2 and prong == 2:
            plt.legend([lam_plt_dict[5], ubst],
                       ['Adv. NN.', 'uBoost'],
                       fontsize=10,
                       frameon=False,
                       loc='upper right'
                       )
        plt.xlim(-0.05, 1.05)
        plt.minorticks_on()
plt.savefig('reports/figures/AugmentTrainingSig.pdf',
            bbox_inches='tight')
