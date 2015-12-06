from __future__ import division
import json
import random
import itertools
import pickle
import numpy as np
import os.path
import sys
from collections import Counter



def gettagset(filename):
    with open(filename,"rb") as datafile:    
        data = json.load(datafile)
        alltags = itertools.chain(*[x["ingredients"] for x in data])
        tag_set = [x[0] for x in Counter(alltags).most_common()]
        # print tagset
        return tag_set

def getfiledata(filename):
    with open(filename,"rb") as datafile:    
        data = json.load(datafile)
        alltags = itertools.chain(*[x["ingredients"] for x in data])
        tag_set = [x[0] for x in Counter(alltags).most_common()]
        bin_data = []
        lbl_data = []
        print "processing file..."
        for i,x in enumerate(data):
            sys.stdout.write("current: {0}/{1}\r".format(i,len(data)))
            sys.stdout.flush()
            bin_data.append(np.array([1 if t in x["ingredients"] else 0 for t in tag_set], dtype="bool"))
            lbl_data.append(x["cuisine"])
        bin_data = np.array(bin_data, dtype="bool")
        # print bin_data 
        # print lbl_data
        # print tag_set
        return bin_data, lbl_data, tag_set
    
def getfilelbl(filename):
    with open(filename,"rb") as datafile:    
        data = json.load(datafile)
        lbl_data = [x["cuisine"] for x in data]
        # print lbl_data
        return lbl_data
    
def loaddata(filename):
    # load bin_data
    objfilename = filename + '.dat'
    if os.path.isfile(objfilename):
        with open(objfilename,'rb') as objfile:
            bin_data = np.load(objfile)
        #load lbl_data
        lbl_data = getfilelbl(filename)
        #load tag_set
        tag_set = gettagset(filename)
    else:
        bin_data, lbl_data, tag_set = getfiledata(filename)
        with open(objfilename,'wb') as objfile:
            np.save(objfile, bin_data)
    
    return bin_data, lbl_data, tag_set

def 

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
    bin_data_pca, dummy1, dummy2 = loaddata(filename)

    bin_data_pca = np.array(bin_data_pca, dtype="float")
    
    print bin_data_pca
    return bin_data_pca

def main():
    bin_data, lbl_data, tag_set = loaddata('train.json')
    k = 10 # k fold
    indices = shuffledata(len(bin_data), k)
    print 'finish processing data'

    bin_data_pca = pca(bin_data)

    nbc(bin_data, lbl_data, indices)
    # end of PCA

if __name__ == "__main__":
    main()