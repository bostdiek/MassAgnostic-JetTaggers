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

# ******************** Overview plot **********************
with open(data_dir + '/Histograms_2p.p', 'rb') as f:
    Hist_2p = pickle.load(f)

with open(data_dir + '/ROCCurves_2p.p', 'rb') as f:
    ROC_2p = pickle.load(f)
print(ROC_2p.keys())


plot_keys = [0.95, 0.9, 0.8, 0.70, 0.60, 0.5]
mycolors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C8']


plt.figure(figsize=(7.25, 3.0))
ax0 = gs0 = gs.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.3)

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
# plt.plot(ROC_2p['GradientBoostingClassifier']['x_data'],
#          ROC_2p['GradientBoostingClassifier']['y_data'],
#          label='BDT',
#          color='C0',
#          ls='--'
#          )
plt.plot(ROC_2p['TauSubjettiness']['x_data'],
         ROC_2p['TauSubjettiness']['y_data'],
         label=r'$\tau_{21}$',
         color='hotpink',
         ls=':'
         )
plt.plot(ROC_2p['AdversaryLambda_050']['x_data'],
         ROC_2p['AdversaryLambda_050']['y_data'],
         label='Adv.',
         color='C4'
         )
plt.plot(ROC_2p['uBoost']['x_data'],
         ROC_2p['uBoost']['y_data'],
         label='uBoost',
         color='blue',
         ls='--'
         )
plt.legend(frameon=False, fontsize=10,
           loc=(0.37, 0.38),
           labelspacing=0.15
           )
# plt.legend(frameon=False, fontsize=10, loc=(0.5, -0.75), ncol=3)
plt.text(0.5, 6e3, '2-prong signal', ha='center', va='top')
plt.minorticks_on()
print('AUC Adv Lamda=50 = {0:0.3f}'.format(ROC_2p['AdversaryLambda_050']['auc']))
print('AUCuBoost = {0:0.3f}'.format(ROC_2p['uBoost']['auc']))
# ***********************************************
gs1 = gs.GridSpecFromSubplotSpec(2, 2, gs0[1], wspace=0.1, hspace=0.1)
ax1 = plt.subplot(gs1[0])
# plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('Arb.')
plt.minorticks_on()
hists = Hist_2p['GradientBoostingClassifier']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, 'BDT',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))
# *****************************************************************
ax2 = plt.subplot(gs1[3], sharex=ax1, sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.minorticks_on()
# plt.ylabel('Arb.')
plt.setp(ax2.get_yticklabels(), visible=False)
hists = Hist_2p['AdversaryLambda_050']
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

# *********************************
ax3 = plt.subplot(gs1[2], sharey=ax1, sharex=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
# plt.setp(ax2.get_yticklabels(), visible=False)
plt.ylabel('Arb.')
plt.minorticks_on()
hists = Hist_2p['uBoost']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, 'uBoost',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))
# *****************************************************************
ax4 = plt.subplot(gs1[1], sharex=ax1, sharey=ax1)
# plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.minorticks_on()
# plt.ylabel('Arb.')
plt.setp(ax4.get_yticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
hists = Hist_2p['BaseNeuralNetwork']
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

fname_name = project_dir.joinpath('reports/figures/AugmentedTrainingTechniques_split.pdf')
plt.savefig(fname_name.resolve(),
            bbox_inches='tight'
            )

# *****************************************************************
# Second Figure            ****************************************
# *****************************************************************
plt.figure(figsize=(8.5, 3.0))
ax0 = gs0 = gs.GridSpec(1, 2, width_ratios=[1, 1.8], wspace=0.3)
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
           loc=(0.4, 0.41),
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
gs1 = gs.GridSpecFromSubplotSpec(2, 3, gs0[1], wspace=0.1, hspace=0.1)
# ***********************************************
ax1 = plt.subplot(gs1[0])
# plt.xlabel(r'$m_{J}$ [GeV]')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.ylabel('Arb.')
plt.minorticks_on()
hists = Hist_2p['TauSubjettiness']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, r'$\tau_{21}$',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))

# ***********************************************
ax2 = plt.subplot(gs1[1], sharey=ax1, sharex=ax1)
# plt.xlabel(r'$m_{J}$ [GeV]')
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.minorticks_on()
hists = Hist_2p['BaseNeuralNetwork']
plt.hist(hists[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hists[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, 'NN',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.4,
         weights=0.2 * np.ones_like(hists[1][0]))

# ***********************************************
ax3 = plt.subplot(gs1[5], sharex=ax1, sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.minorticks_on()
plt.setp(ax3.get_yticklabels(), visible=False)
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
ax4 = plt.subplot(gs1[4], sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.setp(ax4.get_yticklabels(), visible=False)
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
ax5 = plt.subplot(gs1[3], sharey=ax1)
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

# *********************************
fname_name = project_dir.joinpath('reports/figures/AugmentedDataTechniques_split.pdf')
plt.savefig(fname_name.resolve(),
            bbox_inches='tight'
            )
