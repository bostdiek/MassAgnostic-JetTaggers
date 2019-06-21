# -*- coding: utf-8 -*-
'''
Author: Bryan Ostdiek (bostdiek@gmail.com)
This file will calls the individual trainings and saves the results
'''
from pathlib import Path
from datetime import datetime
from subprocess import call
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
datadir = str(project_dir.resolve()) + '/data/'

# Base Neural Network
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_base_nn.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for BaseNN {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('basenn {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')
#
# # Base GBC
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_GBC.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for BaseGBC {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('baseGBC {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')
#
# # PCA Neural Network
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_PCA_nn.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for PCA NN {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('pcann {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')
#
# # Base PCA GBC
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_PCA_GBC.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for PCA GBC {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('pcaGBC {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')

# Planed Neural Network
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_planed_nn.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for Planed NN {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('planednn {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')
#
# # Base Planed GBC
# with open(datadir + 'times.txt', 'a') as fout:
#     for prong in [2, 3, 4]:
#         times = []
#         for i in range(10):
#             start = datetime.now()
#             call('python src/models/train_planed_GBC.py --prong={0} --save=False'.format(prong),
#                  shell=True
#                  )
#             end = datetime.now()
#             times.append(end - start)
#             print('Time for Planed GBC {2}p {0}: {1}'.format(i, end - start, prong))
#         fout.write('planedGBC {0}p:\t'.format(prong))
#         for time in times:
#             fout.write(str(time.total_seconds()))
#             fout.write(',')
#         fout.write(str(np.mean([time.total_seconds() for time in times])))
#         fout.write(',')
#         fout.write(str(np.std([time.total_seconds() for time in times])))
#         fout.write('\n')

# UBOOST
with open(datadir + 'times.txt', 'a') as fout:
    for prong in [3, 4]:
        times = []
        for i in range(10):
            start = datetime.now()
            call('python src/models/train_uBoost.py --prong={0} --save=False'.format(prong),
                 shell=True
                 )
            end = datetime.now()
            times.append(end - start)
            print('Time for uBoost {2}p {0}: {1}'.format(i, end - start, prong))
        fout.write('uBoost {0}p:\t'.format(prong))
        for time in times:
            fout.write(str(time.total_seconds()))
            fout.write(',')
        fout.write(str(np.mean([time.total_seconds() for time in times])))
        fout.write(',')
        fout.write(str(np.std([time.total_seconds() for time in times])))
        fout.write('\n')

# Adversarial
with open(datadir + 'times.txt', 'a') as fout:
    for prong in [3, 4]:
        times = []
        for i in range(10):
            start = datetime.now()
            call('python src/models/train_Adversarial.py --prong={0} --lam_exp=1 --save=False'.format(prong),
                 shell=True
                 )
            end = datetime.now()
            times.append(end - start)
            print('Time for Adversarial {2}p {0}: {1}'.format(i, end - start, prong))
        fout.write('Adversarial {0}p:\t'.format(prong))
        for time in times:
            fout.write(str(time.total_seconds()))
            fout.write(',')
        fout.write(str(np.mean([time.total_seconds() for time in times])))
        fout.write(',')
        fout.write(str(np.std([time.total_seconds() for time in times])))
        fout.write('\n')
