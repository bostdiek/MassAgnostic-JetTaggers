# -*- coding: utf-8 -*-
'''
Authors: Bryan Ostdiek (bostdiek@gmail.com)

This file will reads data from the data/modelpredictions directory
Runs the different metrics for a given prong and saves a dictionary
'''
import click
import numpy as np
import logging
from pathlib import Path
import pickle

# Local packages
from Distances import BhatDist, JS_Distance
from q0_histogram import get_q0_hist


HistConditions = {2: [10000, 100, 35],
                  3: [10000, 100, 35],
                  4: [10000, 100, 35]
                  }


@click.command()
@click.option('--prong', default=2, type=click.IntRange(2, 4),
              help='How many prongs in signal jets')
def run_metrics(prong):
    '''
    '''
    in_name = pred_datadir + 'Histograms_{0}p.p'.format(prong)
    with open(in_name, 'rb') as f:
        AllHists = pickle.load(f)

    N_BACK_O, N_SIG_O, N_BINS = HistConditions[prong]

    MetricsDict = {}
    for model in AllHists:
        print('Computing metrics for {0}'.format(model))
        sig_start, back_start = AllHists[model][1.0]
        SigWeight = N_SIG_O / len(sig_start)
        BackWeight = N_BACK_O / len(back_start)
        print(SigWeight, BackWeight)

        back_hist0, bins = np.histogram(back_start,
                                        range=(50, 400),
                                        bins=N_BINS,
                                        weights=np.ones_like(back_start) * BackWeight
                                        )
        tmp_mets = {'efficiencies': [],
                    'BhatD': [],
                    # 'JSD': [],
                    # 'q0_0.0001': [],
                    # 'q0_0.001': [],
                    # 'q0_0.01': [],
                    # 'q0_0.1': [],
                    # 'q0_0.5': []
                    }

        for eff in AllHists[model]:
            sig, back = AllHists[model][eff]
            sig_h, _ = np.histogram(sig,
                                    bins=bins,
                                    weights=np.ones_like(sig, dtype='float') * SigWeight
                                    )
            back_h, _ = np.histogram(back,
                                     bins=bins,
                                     weights=np.ones_like(back, dtype='float') * BackWeight
                                     )
            # print(eff)
            # print(sig_h, back_h)
            BD = BhatDist(back_hist0, back_h)
            # JS = JS_Distance(back_hist0, back_h)
            # q0_0001 = get_q0_hist(sig_hist=sig_h,
            #                       back_hist=back_h,
            #                       uncert=0.0001,
            #                       back_weight=BackWeight
            #                       )
            # # print(q0_0001)
            # q0_001 = get_q0_hist(sig_hist=sig_h,
            #                      back_hist=back_h,
            #                      uncert=0.001,
            #                      back_weight=BackWeight
            #                      )
            # q0_010 = get_q0_hist(sig_hist=sig_h,
            #                      back_hist=back_h,
            #                      uncert=0.010,
            #                      back_weight=BackWeight
            #                      )
            # q0_100 = get_q0_hist(sig_hist=sig_h,
            #                      back_hist=back_h,
            #                      uncert=0.10,
            #                      back_weight=BackWeight
            #                      )
            # q0_500 = get_q0_hist(sig_hist=sig_h,
            #                      back_hist=back_h,
            #                      uncert=0.5,
            #                      back_weight=BackWeight
            #                      )

            tmp_mets['efficiencies'].append(eff)
            tmp_mets['BhatD'].append(BD)
            # tmp_mets['JSD'].append(JS)
            # tmp_mets['q0_0.0001'].append(q0_0001)
            # tmp_mets['q0_0.001'].append(q0_001)
            # tmp_mets['q0_0.01'].append(q0_010)
            # tmp_mets['q0_0.1'].append(q0_100)
            # tmp_mets['q0_0.5'].append(q0_500)

        MetricsDict[model] = tmp_mets

    outname = pred_datadir + 'Metrics_{0}p.p'.format(prong)
    with open(outname, 'wb') as fout:
        pickle.dump(MetricsDict, fout)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    pred_datadir = str(project_dir.resolve()) + '/data/modelpredictions/'
    run_metrics()
