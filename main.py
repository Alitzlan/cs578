from __future__ import division
import json
import random
import itertools
import pickle
import numpy as np
import os.path
from collections import Counter
from sklearn.decomposition import PCA

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
    
def main():
    bin_data, lbl_data = readdata('train.json')
    indices = shuffledata(len(bin_data), 10)
    print 'finish processing data'

    # Added for PCA
    print "Starts doing PCA"
    num_parameter = len(bin_data[0])
    pca = PCA(n_components = num_parameter)
    pca.fit(bin_data)
    print(pca.explained_variance_ratio_)
    bin_data_pca = np.multiply(bin_data, pca.components_)
    print bin_data_pca
    # end of PCA

    
if __name__ == "__main__":
    main()