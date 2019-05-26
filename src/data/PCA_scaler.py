# -*- coding: utf-8 -*-
'''
Author: Andrea Mitridate; Bryan Ostdiek (bostdiek@gmail.com)
This file saves data to the data/interm directory
First bins the data, rotates each bin to the PCA basis, scales, then rotates back
'''
import numpy as np
import numpy.random as ra
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd


class PCA_scaler_withRotations:
    def __init__(self, jet_mass, data, bins, minmass=50, maxmass=400):
        '''
        Bins the data, rotates each bin to the PCA basis, scales, then rotates back
        Input:
            jet_mass: (np.array), data to be binned
            data: (np.array), actual training data which will be scaled and rotated
            bins: (int) number of bins to use. Bin sizes will be chosen to give
                equal number of jevents per bin
        '''
        self.bins = bins
        eventsperbin = int(len(data) / bins)
        self.bin_edges = [minmass]
        sorted_jet_mass = np.sort(jet_mass)
        for i in range(1, bins):
            self.bin_edges.append(sorted_jet_mass[i * eventsperbin])
        self.bin_edges.append(maxmass)
        assert len(self.bin_edges) == bins + 1

        bin_inds = np.digitize(jet_mass, self.bin_edges) - 1

        # get a standard scaler for each bin
        self.initial_scale = []
        tmp_data = np.zeros_like(data)
        for i in range(bins):
            scaler = StandardScaler().fit(data[bin_inds == i])
            self.initial_scale.append(scaler)
            # scale the data
            tmp_data[bin_inds == i] = scaler.transform(data[bin_inds == i])

        # find the rotations for the qcd scaled data to the PCA basis
        self.mypca = []
        tmp_data_pca = np.zeros_like(data)
        for i in range(bins):
            K = tmp_data_pca.T.dot(tmp_data) / tmp_data.shape[0]  # covariance matrix
            u, s, v = svd(K)  # u, s, v decomposition
            self.mypca.append(u)
            tmp_data_pca[bin_inds == i] = tmp_data[bin_inds == i].dot(u)

        # finally, perform one last scaling of the now PCA rotated data
        self.final_scale = []
        for i in range(bins):
            scaler = StandardScaler().fit(data[bin_inds == i])
            self.final_scale.append(scaler)

    def transform(self, jet_mass, data):
        '''
        Inputs:
            jet_mass: (np.array) mass to be binned
            data: (np.array) data to be scaled
        Outputs:
            scaled data ready to be fed to network
        '''
        bin_inds = np.digitize(jet_mass, self.bin_edges) - 1
        scaled = np.zeros_like(data)
        for i in range(self.bins):
            tmp_data = data[bin_inds == i]
            initial_scaler = self.initial_scale[i]
            pca_u = self.mypca[i]
            final_scaler = self.final_scale[i]

            tmp_data = initial_scaler.transform(tmp_data)  # scale
            tmp_data = tmp_data.dot(pca_u)  # rotate
            tmp_data = final_scaler.transform(tmp_data)  # scalle
            tmp_data = tmp_data.dot(pca_u.T)  # rotate back

            scaled[bin_inds == i] = tmp_data
        return scaled


class PCA_scaler:
    def __init__(self, bkgdata, bins):
        eventxbin = len(bkgdata) / bins
        bkgdata = bkgdata[bkgdata[:, 0].argsort()]
        bkgdatabinned = []
        self.binboundaries = []
        for i in range(0, bins):
            bkgdatabinned.append(bkgdata[i * eventxbin: (i + 1) * eventxbin, 2:])
            self.binboundaries.append(bkgdata[i * eventxbin, 0])
        self.scale1 = [preprocessing.StandardScaler().fit(databin)
                       for databin in bkgdatabinned]
        tempdata = [self.scale1[i].transform(databin)
                    for i, databin in enumerate(bkgdatabinned)]
        self.mypca = [PCA().fit(tempdatarow)
                      for tempdatarow in tempdata]
        newtempdata = [self.mypca[i].transform(tempdatarow)
                       for i, tempdatarow in enumerate(tempdata)]
        self.scale2 = [preprocessing.StandardScaler().fit(tempdatarow)
                       for tempdatarow in newtempdata]

    def transform(self, inputdata):
        inputdatabinned = []
        inputmassbinned = []
        for i in range(0, len(self.binboundaries) - 1):
            inputdatabinned.append(
                np.array([myrow[2:] for myrow in inputdata
                         if (myrow[0] < self.binboundaries[i + 1] and myrow[0] >= self.binboundaries[i])])
            )
            inputmassbinned.append(
                np.array([myrow[0:1] for myrow in inputdata
                         if (myrow[0] < self.binboundaries[i + 1] and myrow[0] >= self.binboundaries[i])])
            )
        scaleddatabinned = [np.insert(np.dot(self.scale2[i].transform(self.mypca[i].transform(self.scale1[i].transform(inputdatarow))),
                                             self.mypca[i].components_), [0], inputmassbinned[i], axis=1)
                            if len(inputdatarow) > 0 else []
                            for i, inputdatarow in enumerate(inputdatabinned)]
        return scaleddatabinned
