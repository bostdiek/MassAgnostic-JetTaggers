print "This program uses uBoost to study mass decorrelation"

import sys
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import callbacks as cb
from sklearn import preprocessing, model_selection, metrics
import numpy as np
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
#from imblearn.datasets import make_imbalance
from scipy.stats import itemfreq
from collections import OrderedDict
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from HelperFunctions import ReadPickle, WritePickle, ReadPickleHist, WritePickleHist

import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# this wrapper makes it possible to train on subset of features
from rep.estimators import SklearnClassifier
from hep_ml.commonutils import train_test_split
from hep_ml import uboost, gradientboosting as ugb, losses

import random
from rep.metaml import ClassifiersFactory
from rep.report.metrics import RocAuc

def GetThresholdForGivenEfficiency(proba_array, eff):
    thresholds = np.linspace(0,1,100)
    for iter_threhsold in thresholds:
        if (np.divide(len(np.argwhere(proba_array > iter_threhsold)), len(proba_array), dtype = 'float') < eff):
            return iter_threhsold


data_from_sig_file = np.loadtxt('data_sig.txt', delimiter=',')
data_from_bkg_file = np.loadtxt('data_bkg.txt', delimiter=',')


# Put a fake entry = 11.0, on which I can use uBoost to uniformize, hence doing nothing. 
data_from_sig_file = np.append(data_from_sig_file, np.full([data_from_sig_file.shape[0], 1],11.0), axis = 1)
data_from_bkg_file = np.append(data_from_bkg_file, np.full([data_from_bkg_file.shape[0], 1],11.0), axis = 1)

data_from_sig_file = np.append(data_from_sig_file, np.ones([data_from_sig_file.shape[0], 1],'int'), axis = 1)
data_from_bkg_file = np.append(data_from_bkg_file, np.zeros([data_from_bkg_file.shape[0], 1],'int'), axis = 1)
data_all = np.append(data_from_sig_file, data_from_bkg_file, axis = 0)
data = pandas.DataFrame(data_all, columns = ['(pruned)m', 'pT', 'tau_(1)^(1/2)','tau_(1)^(1)','tau_(1)^(2)','tau_(2)^(1/2)','tau_(2)^(1)','tau_(2)^(2)','tau_(3)^(1/2)','tau_(3)^(1)','tau_(3)^(2)','tau_(4)^(1)','tau_(4)^(2)', 'fake_feature', 'labels']) 

#print data
labels = data['labels']

uniform_features_none  = ['fake_feature']
uniform_features_mass  = ['(pruned)m']
train_features = ['tau_(1)^(1/2)','tau_(1)^(1)','tau_(1)^(2)','tau_(2)^(1/2)','tau_(2)^(1)', 'tau_(2)^(2)','tau_(3)^(1/2)','tau_(3)^(1)','tau_(3)^(2)','tau_(4)^(1)','tau_(4)^(2)']
n_estimators = 150
base_estimator = DecisionTreeClassifier(max_depth=4)

trainX, testX, trainY, testY = train_test_split(data, labels, random_state=42, train_size=0.7)

classifiers = ClassifiersFactory()

uboost_clf_uniform_mass = uboost.uBoostClassifier(uniform_features=uniform_features_mass, uniform_label=0,
                                     base_estimator=base_estimator, train_features=train_features
                                     ,n_estimators=n_estimators
                                     #, efficiency_steps=12, n_threads=4
                                     )
#uboost_clf_uniform_none = uboost.uBoostClassifier(uniform_features=uniform_features_none, uniform_label=0,
#                                     base_estimator=base_estimator,train_features=train_features
#                                     ,n_estimators=n_estimators
                                     #, efficiency_steps=12, n_threads=4
#                                     )
base_ada = GradientBoostingClassifier(max_depth=4, n_estimators=n_estimators, learning_rate=0.1)
classifiers['AdaBoost']         = SklearnClassifier(base_ada, features=train_features)
#classifiers['uniform_none']     = SklearnClassifier(uboost_clf_uniform_none)
classifiers['uniform_mass']     = SklearnClassifier(uboost_clf_uniform_mass)

fit_result = classifiers.fit(trainX, trainY, parallel_profile='threads-4')
pass


report = classifiers.test_on(testX, testY)
#print report.efficiencies(features=None)
plt.figure()
#plt.ylim(0.88, 0.94)
result = report.learning_curve(RocAuc(), steps=1)
result.plot()
plt.savefig("metrics.pdf")


testX_sig = testX[testX['labels'] == 1]
testX_bkg = testX[testX['labels'] == 0]
prediction_sig  = classifiers.predict_proba(testX_sig)
prediction_bkg  = classifiers.predict_proba(testX_bkg)


prediction_full = classifiers.predict_proba(testX)

plt.figure()
#rep.report.classification.ClassificationReport(fit_result, testX)
sig_effs = np.linspace(0.05,1,96) 
dict_saveData = OrderedDict()
for iter_classifiers in classifiers.keys():
    tagger_on_sig  = prediction_sig[iter_classifiers][:,1]
    tagger_on_bkg  = prediction_bkg[iter_classifiers][:,1]
    tagger_on_full = prediction_full[iter_classifiers][:,1]
    dict_temp = OrderedDict()

    for eff in sig_effs:
        threshold = GetThresholdForGivenEfficiency(tagger_on_full, eff) 
        indices_full = np.argwhere(tagger_on_full > threshold)

        indices_sig = np.argwhere(tagger_on_sig > threshold)
        indices_bkg = np.argwhere(tagger_on_bkg > threshold)
        mass_sig = np.array(testX_sig)[:,0]
        mass_bkg = np.array(testX_bkg)[:,0]
        mass_sig = mass_sig[indices_sig]
        mass_bkg = mass_bkg[indices_bkg]
        mass_sig = mass_sig.flatten()
        mass_bkg = mass_bkg.flatten()
        #print mass_bkg 
        plt.hist(mass_sig, bins=np.linspace(50,400,20), histtype = 'step', rwidth=0.85, alpha=0.7, normed=1, label='uBoost_' + iter_classifiers + '_sig:eff=' + str(eff))
        plt.hist(mass_bkg, bins=np.linspace(50,400,20), histtype = 'step', rwidth=0.85, alpha=0.7, normed=1, label='uBoost_' + iter_classifiers + '_bkg:eff=' + str(eff))
        dict_temp[eff] = (mass_sig, mass_bkg)
    dict_saveData[iter_classifiers] = dict_temp
        #WritePickleHist(mass_sig, mass_bkg, 'hist_uBoost_' + iter_classifiers + '_eff:' + str(eff) + '.p')

WritePickleHist(dict_saveData, 'data_hist.p')

plt.legend(loc='upper right', fontsize = 'x-small')
plt.savefig('hist.pdf')

plt.figure()
gridwidth   = 0.5
dashes      = [8,4]
dotdashes   = [8,4,1,4]
delta       = 0.025
x           = np.arange(0, 1.0+delta, delta)
logy        = np.arange(0,math.log(1e4), delta)
y           = np.exp(logy)
X, Y        = np.meshgrid(x, y)
Z           = np.multiply(X,np.power(Y,0.5))
gridwidth   = 0.5
CS          = plt.contour(X, Y, Z,[1,2,4,8], colors='gray',linestyles='dashed',linewidths=gridwidth)
manual_loc  = [(0.8,1/(0.8*0.8)),(0.8,4/(0.8*0.8)), (0.8,16/(0.8*0.8)), (0.8,64/(0.8*0.8))]

for c in CS.collections:
    c.set_dashes([(0,(8,4))])

plt.clabel(CS, inline=1, fontsize=10,fmt='%3.0f',manual=manual_loc)
result = report.roc()
for name, data_xy in result.functions.items():
    x_val, y_val = data_xy
    plt.plot(y_val, np.divide(1, x_val), linewidth=1, label=name)
    WritePickle(y_val, np.divide(1, x_val), "roc_" + name + ".p") 
plt.yscale('log')
plt.grid(True)
plt.axis([0, 1, 1, 10000])
plt.legend(loc='upper right')
plt.savefig("roc.pdf")

