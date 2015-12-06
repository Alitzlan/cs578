from __future__ import division
import json
import random
import itertools
import pickle
import numpy as np
import os.path
from collections import Counter



def processfile(filename):
    with open(filename) as datafile:    
        data = json.load(datafile)
        alltags = itertools.chain(*[x["ingredients"] for x in data])
        lbl_data = [x["cuisine"] for x in data]
        tagset_c = Counter(alltags).most_common()
        tagset = [x[0] for x in tagset_c]
        bin_data = np.array([np.array([1 if t in x["ingredients"] else 0 for t in tagset]) for x in data])
        print bin_data
        return bin_data, lbl_data
    
def shuffledata(datalen, numpart):
    idx = range(0, datalen)
    random.shuffle(idx)
    step = int(datalen/numpart)
    parts = []
    for i in range(0,numpart):
        parts.append(idx[i*step:(i+1)*step])
    return parts
    
def readdata(filename):
    objfilename = filename + '.dat'
    if os.path.isfile(objfilename):
        with open(objfilename, "rb") as objfile:
            bin_data, lbl_data = pickle.load(objfile)
    else:
        bin_data, lbl_data = processfile(filename)
        with open(objfilename, "wb") as objfile:
            pickle.dump([bin_data, lbl_data], objfile)
    return bin_data, lbl_data


def nbc(bin_data, lbl_data, indices):
    print "Starts nbc"
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    k = len(indices)
    test_indices = []
    train_indices = []
    for i in xrange(k):
        test_indices = []
        train_indices = []
        test_indices.extend(indices[i])
        for j in xrange(k):
            if j!=i:
                train_indices.extend(indices[j])
        # print test_indices
        # print train_indices
        train_data = np.take(bin_data, train_indices, axis=0)
        train_lbl_data = np.take(lbl_data, train_indices, axis=0)

        test_data = np.take(bin_data, test_indices, axis=0)
        test_lbl_data = np.take(lbl_data, test_indices, axis=0)

        # print train_data
        # print test_data
        gnb.fit(train_data, train_lbl_data)
        predicted = gnb.predict(test_data)

        # print test_lbl_data
        # print predicted
        print ("Number of mislabeled points out of a total %d points : %d" % (test_data.shape[0], (test_lbl_data != predicted).sum()))

def pca(bin_data):
    from sklearn.decomposition import PCA
    # Added for PCA
    print "Starts doing PCA"
    num_parameter = len(bin_data[0])
    print "Originally " + str(num_parameter) + " parameters"
    filename = 'train.json'
    objfilename = filename + '.pca.dat'
    if os.path.isfile(objfilename):
        with open(objfilename, "rb") as objfile:
            bin_data_pca = pickle.load(objfile)
    else:
        pca = PCA(n_components = num_parameter)
        pca.fit(bin_data)
        print(pca.explained_variance_ratio_)
        epsilon = 0.95
        accum_epsilon = 0.0
        for i in range(len(pca.explained_variance_ratio_)):
            accum_epsilon = accum_epsilon + pca.explained_variance_ratio_[i]
            if accum_epsilon > epsilon:
                break
        new_num_parameter = i # m dimensional
        print "after PCA, " + str(new_num_parameter) + " parameters"
        print pca.components_[0:new_num_parameter, ]

        bin_data_pca = []
        for j in xrange(len(bin_data)):
            x_j = []
            for i in xrange(len(pca.components_[0:new_num_parameter, ])):
                x_j_i = np.dot(pca.components_[i], bin_data[j])
                x_j.append(x_j_i)
            bin_data_pca.append(x_j)
        bin_data_pca = np.array(bin_data_pca)
        with open(objfilename, "wb") as objfile:
            pickle.dump([bin_data_pca], objfile)

    print bin_data_pca
    return bin_data_pca

def main():
    bin_data, lbl_data = readdata('train.json')
    k = 10 # k fold
    indices = shuffledata(len(bin_data), k)
    print 'finish processing data'

    # bin_data_pca = pca(bin_data)

    nbc(bin_data, lbl_data, indices)
    # end of PCA


    #NBC



if __name__ == "__main__":
    main()