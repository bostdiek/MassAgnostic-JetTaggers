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

# Load data
project_dir = Path(__file__).resolve().parents[2]
p_name = project_dir.joinpath('data/modelpredictions/')
data_dir = str(p_name.resolve())
with open(data_dir + '/Histograms_2p.p', 'rb') as f:
    Hist_2p = pickle.load(f)

with open(data_dir + '/ROCCurves_2p.p', 'rb') as f:
    ROC_2p = pickle.load(f)


# ************* Plot *************
plt.figure(figsize=(8, 2.5))
gs0 = gs.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.3)
plot_keys = [0.95, 0.9, 0.8, 0.70, 0.60, 0.5]
mycolors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']

plt.subplot(gs0[0])
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.yscale('log')
plt.ylim(1, 1e4)
plt.xlim(0, 1)
plt.plot(ROC_2p['BaseNeuralNetwork']['x_data'],
         ROC_2p['BaseNeuralNetwork']['y_data'],
         label='NN: ' + '{0:.3f}'.format(ROC_2p['BaseNeuralNetwork']['auc'])
         )
plt.plot(ROC_2p['GradientBoostingClassifier']['x_data'],
         ROC_2p['GradientBoostingClassifier']['y_data'],
         label='GBC: ' + '{0:.3f}'.format(ROC_2p['GradientBoostingClassifier']['auc'])
         )
plt.legend(frameon=False, fontsize=12, loc=(0.18, 0.55))
plt.text(0.5, 6e3, '2-Prong Signal', ha='center', va='top')
plt.minorticks_on()

gs1 = gs.GridSpecFromSubplotSpec(1, 2, gs0[1], wspace=0.1)
ax1 = plt.subplot(gs1[0])
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylabel('Arb.')
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
plt.text(450 / 2, 7e3, 'Neural Network', fontsize=12, ha='center', va='top')

plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.5,
         weights=0.2 * np.ones_like(hists[1][0]))

ax2 = plt.subplot(gs1[1], sharey=ax1)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.setp(ax2.get_yticklabels(), visible=False)
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
plt.text(450 / 2, 7e3, 'Gradient Boosted\nDecision Tree',
         fontsize=12, ha='center', va='top')
plt.hist(hists[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.5,
         weights=0.2 * np.ones_like(hists[1][0]))

fname_name = project_dir.joinpath('reports/figures/TraditionalTechniques.pdf')
plt.savefig(fname_name.resolve(),
            bbox_inches='tight'
            )
