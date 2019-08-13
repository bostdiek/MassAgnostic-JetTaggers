import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Patch
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

# ******************** Overview plot **********************
with open(data_dir + '/Histograms_2p.p', 'rb') as f:
    Hist_2p = pickle.load(f)

with open(data_dir + '/ROCCurves_2p.p', 'rb') as f:
    ROC_2p = pickle.load(f)

plt.figure(figsize=(8.5, 2.0))
ax0 = gs0 = gs.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.3)
plot_keys = [0.95, 0.9, 0.8, 0.70, 0.60, 0.5]
mycolors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C8']
# ************* Plot *************
plt.subplot(gs0[0])
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.yscale('log')
plt.ylim(1, 1e4)
plt.yticks([1, 10, 100, 1000, 10000])
plt.xlim(0, 1)
plt.plot(ROC_2p['BaseNeuralNetwork']['x_data'],
         ROC_2p['BaseNeuralNetwork']['y_data'],
         label='Orig.',
         color='C0',
         ls='-'
         )
plt.plot(ROC_2p['TauSubjettiness']['x_data'],
         ROC_2p['TauSubjettiness']['y_data'],
         label=r'$\tau_{21}$',
         color='hotpink',
         ls=':'
         )
plt.plot(ROC_2p['PCANeuralNetwork']['x_data'],
         ROC_2p['PCANeuralNetwork']['y_data'],
         label='PCA',
         color='C3'
         )
plt.plot(ROC_2p['PlanedNeuralNetwork']['x_data'],
         ROC_2p['PlanedNeuralNetwork']['y_data'],
         label='Planed',
         color='C2'
         )
# plt.legend(frameon=False, fontsize=12, loc=(0.3, 0.4))
plt.plot(ROC_2p['TauDDT']['x_data'],
         ROC_2p['TauDDT']['y_data'],
         label=r'$\tau_{21}^{\rm{DDT}}$',
         color='purple',
         ls=':'
         )
plt.legend(frameon=False, fontsize=10,
           loc=(0.35, 0.31),
           labelspacing=0.15
           )
# plt.legend(frameon=False, fontsize=10, loc=(0.5, -0.75), ncol=3)
plt.text(0.5, 6e3, '2-prong signal', ha='center', va='top')
plt.minorticks_on()
print('AUC NN (base) = {0:0.3f}'.format(ROC_2p['BaseNeuralNetwork']['auc']))
print('AUC NN (PCA ) = {0:0.3f}'.format(ROC_2p['PCANeuralNetwork']['auc']))
print('AUC NN (Plan) = {0:0.3f}'.format(ROC_2p['PlanedNeuralNetwork']['auc']))
print('AUC tau_21prime = {0:0.3f}'.format(ROC_2p['TauDDT']['auc']))
# ***********************************************
gs1 = gs.GridSpecFromSubplotSpec(1, 3, gs0[1], wspace=0.1)
ax1 = plt.subplot(gs1[2])
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.minorticks_on()
plt.setp(ax1.get_yticklabels(), visible=False)
hists = Hist_2p['PCANeuralNetwork']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, 'PCA', fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))

# *********************************
ax2 = plt.subplot(gs1[1], sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.minorticks_on()
hists = Hist_2p['PlanedNeuralNetwork']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, 'Planed',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))

# *********************************
ax2 = plt.subplot(gs1[0], sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.ylabel('Arb.')
plt.minorticks_on()
hists = Hist_2p['TauDDT']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, r'$\tau_{21}^{\rm{DDT}}$',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))

fname_name = project_dir.joinpath('reports/figures/AugmentedDataTechniques.pdf')
plt.savefig(fname_name.resolve(),
            bbox_inches='tight'
            )
# ***************************************************
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

    plt.plot(ROCS['BaseNeuralNetwork']['x_data'],
             ROCS['BaseNeuralNetwork']['y_data'],
             color='C0'
             )
    plt.plot(ROCS['PCANeuralNetwork']['x_data'],
             ROCS['PCANeuralNetwork']['y_data'],
             color='C3'
             )
    plt.plot(ROCS['PlanedNeuralNetwork']['x_data'],
             ROCS['PlanedNeuralNetwork']['y_data'],
             color='C2'
             )
    taunn, = plt.plot(ROCS['TauSubjettiness']['x_data'],
                      ROCS['TauSubjettiness']['y_data'],
                      color='hotpink',
                      ls=':',
                      zorder=10
                      )
    if prong == 2:
        taunnddt, = plt.plot(ROCS['TauDDT']['x_data'],
                             ROCS['TauDDT']['y_data'],
                             color='purple',
                             ls=':',
                             zorder=10
                             )
    plt.plot(ROCS['GradientBoostingClassifier']['x_data'],
             ROCS['GradientBoostingClassifier']['y_data'],
             color='C0', ls='--'
             )
    plt.plot(ROCS['PCAGBC']['x_data'],
             ROCS['PCAGBC']['y_data'],
             color='C3', ls='--'
             )
    plt.plot(ROCS['PlanedGBC']['x_data'],
             ROCS['PlanedGBC']['y_data'],
             color='C2', ls='--'
             )
    nn, = plt.plot([], [], color='k', label='NN')
    gbc, = plt.plot([], [], color='k', ls='--', label='BDT')
    singlevar, = plt.plot([], [], color='k', ls=':', label='Single Variable')
    if prong == 2:
        plt.ylabel('Background Rejection')
        # plt.ylabel('Bhat. Dist.')
        plt.legend([Patch(facecolor='C0',
                          label='Original'),
                    Patch(facecolor='C3',
                          label='PCA'),
                    Patch(facecolor='C2',
                          label='Planed'), taunn, taunnddt],
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

    plt.plot(AllMets['BaseNeuralNetwork']['efficiencies'],
             AllMets['BaseNeuralNetwork']['BhatD'],
             color='C0'
             )
    plt.plot(AllMets['PCANeuralNetwork']['efficiencies'],
             AllMets['PCANeuralNetwork']['BhatD'],
             color='C3'
             )
    plt.plot(AllMets['PlanedNeuralNetwork']['efficiencies'],
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
        plt.legend([Patch(facecolor='C0',
                          label='Original'),
                    Patch(facecolor='C3',
                          label='PCA'),
                    Patch(facecolor='C2',
                          label='Planed'),
                    taunn, taunnddt],
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

plt.savefig('reports/figures/AugmentDataBhat.pdf',
            bbox_inches='tight')
plt.clf()
plt.close()

# *********** Background rejection versus Distance ***********
plt.figure(figsize=(8.5, 2.5))
gs0 = gs.GridSpec(1, 3, wspace=0.1)

for prong in [2, 3, 4]:
    if prong == 2:
        ax = plt.subplot(gs0[prong - 2])
        plt.ylabel('Bhattacharyya Distance')
    else:
        ax1 = plt.subplot(gs0[prong - 2])
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
        plt.scatter(backrej(0.5), dist(0.5), marker='*', s=25, color='purple',
                    zorder=10)
        plt.scatter(backrej(0.25), dist(0.25), marker='s', s=25, color='purple',
                    zorder=10)
        plt.scatter(backrej(0.75), dist(0.75), marker='o', s=25, color='purple',
                    zorder=10)

    plt.title('{0}-prong'.format(prong))
    if prong == 2:
        plt.ylabel('Bhattacharyya Distance')
        # plt.ylabel('Bhat. Dist.')
        base = plt.scatter([], [], color='C0', marker='s')
        pca = plt.scatter([], [], color='C3', marker='s')
        planed = plt.scatter([], [], color='C2', marker='s')
        plt.legend([Patch(facecolor='C0',
                          label='Original'),
                    Patch(facecolor='C3',
                          label='PCA'),
                    Patch(facecolor='C2',
                          label='Planed'),
                    taunn, taunnddt],
                   ['Original', 'PCA', 'Planed', r'$\tau_N / \tau_{N-1}$',
                    r'$\tau_{21}^{\rm{DDT}}$'],
                   fontsize=10,
                   frameon=False,
                   labelspacing=0.15,
                   ncol=2,
                   columnspacing=1,
                   handlelength=1.5
                   )
        ax.annotate("Better",
                    ha='center',
                    fontsize=10,
                    xy=(1e3, 0.),
                    xytext=(1e3, 0.35),
                    rotation=90,
                    arrowprops=dict(arrowstyle="->")
                    )
        ax.annotate("Better",
                    va='center',
                    fontsize=10,
                    xy=(1e3, 0.),
                    xytext=(1e2, 0.0),
                    arrowprops=dict(arrowstyle="->")
                    )
        plt.xlim(2e3, 0.8)
    if prong == 3:
        plt.legend([nn, gbc],  # , singlevar],
                   ['NN', 'BDT'],  # , 'Single Variable'],
                   fontsize=10,
                   frameon=False)
        plt.xlim(2e3, 0.8)
    if prong == 4:
        eff50 = plt.scatter([], [], marker='*', s=25, color='k', zorder=10)
        eff25 = plt.scatter([], [], marker='s', s=25, color='k', zorder=10)
        eff75 = plt.scatter([], [], marker='o', s=25, color='k', zorder=10)
        plt.legend([eff75, eff50, eff25],
                   [r'$\varepsilon_{S} = 0.75$',
                    r'$\varepsilon_{S} = 0.50$',
                    r'$\varepsilon_{S} = 0.25$'
                    ],
                   frameon=False,
                   fontsize=10,
                   loc='upper right'
                   )
plt.savefig('reports/figures/AugmentedDataBackgroundRejectionVersusDistance.pdf',
            bbox_inches='tight')
