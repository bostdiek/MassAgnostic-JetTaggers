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

lam_exp_list = ['0',  # '3.010e-01', '6.990e-01',
                '1',  # '1.301e+00', '1.699e+00',
                '2',  # '2.301e+00', '2.699e+00']  # , '3'
                ]

plt.figure(figsize=(8.5, 2.0))
ax0 = gs0 = gs.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.3)
plot_keys = [0.95, 0.9, 0.8, 0.70, 0.60, 0.5]
mycolors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
# ***********************************************
lam_colors = ['C0',  # 'C0', 'C0',
              'C1',  # 'C1', 'C1',
              'hotpink',  # 'hotpink', 'hotpink'
              ]
lam_ls = ['solid', ':', '--']  # , 'solid', ':', '--', 'solid', ':', '--']
for i, le in enumerate(lam_exp_list):
    lam = 10**float(le)
    if lam > 0:
        lam = round(lam)
    frocname = data_dir + 'adv_nn_lam_{0:03d}_2p_roc.p'.format(lam)
    with open(frocname, 'rb') as f:
        roc_dict = pickle.load(f)
    plt.subplot(gs0[0])
    plt.plot(roc_dict['x_data'],
             roc_dict['y_data'],
             label=r'$\lambda=$' + str(lam),  # + ': {0:.3f}'.format(roc_dict['auc'])
             # ls=lam_ls[i],
             color=lam_colors[i]
             )

plt.subplot(gs0[0])
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.yscale('log')
plt.ylim(1, 1e4)
plt.xlim(0, 1)
plt.legend(frameon=False, fontsize=10, loc=(0.3, 0.45))
# plt.legend(frameon=False, fontsize=10, loc=(0.5, -0.75), ncol=3)
plt.text(0.5, 6e3, '2-Prong Signal', ha='center', va='top')
plt.minorticks_on()

# ***********************************************
lam = 1
gs1 = gs.GridSpecFromSubplotSpec(1, 3, gs0[1], wspace=0.1)
ax1 = plt.subplot(gs1[0])

fhistname = data_dir + 'adv_nn_lam_{0:03d}_2p_histos.p'.format(lam)
with open(fhistname, 'rb') as f:
    hist_dict = pickle.load(f)

plt.hist(hist_dict[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hist_dict[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, r'$\lambda=$' + str(lam), fontsize=12, ha='center', va='top')

plt.hist(hist_dict[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.5,
         weights=0.2 * np.ones_like(hist_dict[1][0]))
plt.xlim(50, 400)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.minorticks_on()
plt.ylabel('Arb.')

# ***********************************************
ax2 = plt.subplot(gs1[1], sharey=ax1)
lam = 10

fhistname = data_dir + 'adv_nn_lam_{0:03d}_2p_histos.p'.format(lam)
with open(fhistname, 'rb') as f:
    hist_dict = pickle.load(f)

plt.hist(hist_dict[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hist_dict[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, r'$\lambda=$' + str(lam), fontsize=12, ha='center', va='top')

plt.hist(hist_dict[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.5,
         weights=0.2 * np.ones_like(hist_dict[1][0]))
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.setp(ax2.get_yticklabels(), visible=False)
plt.minorticks_on()

# ***********************************************
ax3 = plt.subplot(gs1[2], sharey=ax1)
lam = 100

fhistname = data_dir + 'adv_nn_lam_{0:03d}_2p_histos.p'.format(lam)
with open(fhistname, 'rb') as f:
    hist_dict = pickle.load(f)

plt.hist(hist_dict[1][1], range=(50, 400), histtype='step', bins=35, color='k')
for i, eff in enumerate(plot_keys):
    plt.hist(hist_dict[eff][1],
             range=(50, 400),
             histtype='step',
             bins=35,
             color=mycolors[i]
             )
plt.text(450 / 2, 7e3, r'$\lambda=$' + str(lam), fontsize=12, ha='center', va='top')

plt.hist(hist_dict[1][0], range=(50, 400),
         bins=35,
         color='grey',
         alpha=0.5,
         weights=0.2 * np.ones_like(hist_dict[1][0]))
plt.xlim(50, 400)
plt.yscale('log')
plt.ylim(10, 1e4)
plt.xlabel(r'$m_{J}$ [GeV]')
plt.setp(ax3.get_yticklabels(), visible=False)
plt.minorticks_on()

# ***********************************************
plt.savefig('reports/figures/AdversaryShapes.pdf', bbox_inches='tight')
