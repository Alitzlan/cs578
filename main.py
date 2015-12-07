from __future__ import division
import json
import random
import itertools
import pickle
import numpy as np
import os.path
import sys
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from collections import Counter
from sklearn.decomposition import PCA

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
        return np.array(bin_data, dtype="float"), lbl_data, tag_set
    
def getfilelbl(filename):
    with open(filename,"rb") as datafile:    
        data = json.load(datafile)
        lbl_data = [x["cuisine"] for x in data]
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
    
def shuffledata(datalen, numpart):
    idx = range(0, datalen)
    random.shuffle(idx)
    step = int(datalen/numpart)
    parts = []
    for i in range(0,numpart):
        parts.append(idx[i*step:(i+1)*step])
    return parts

def subset(bin_data, lbl_data, start=0, end=-1):
    return bin_data[start:end], lbl_data[start:end]
    
def main():
    bin_data, lbl_data, tag_set = loaddata('train.json')
    tbin_data, tlbl_data = subset(bin_data, lbl_data, start=5000, end=6000)
    bin_data, lbl_data = subset(bin_data, lbl_data, end=5000)
    indices = shuffledata(len(bin_data), 10)
    # restore format
    bin_data = np.array(bin_data,dtype="float")
    print 'finish processing data'

    svd = TruncatedSVD(3000)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    bin_data = lsa.fit_transform(bin_data)
    logistic = linear_model.LogisticRegression(C=100000)
    rbm = BernoulliRBM(verbose=True, learning_rate=0.001)
    classifier = Pipeline(steps=[('rbm', rbm),('logistic', logistic)])
    classifier.fit(bin_data, lbl_data)
    print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        tlbl_data,
        classifier.predict(lsa.transform(tbin_data)))))
    print classifier.predict(lsa.transform(tbin_data))

#     # Added for PCA
#     print "Starts doing PCA"
#     num_parameter = len(bin_data[0])
#     print "Originally " + str(num_parameter) + " parameters"
#     pca = PCA(n_components = num_parameter)
#     pca.fit(bin_data)
#     print(pca.explained_variance_ratio_)
#     epsilon = 0.95
#     accum_epsilon = 0.0
#     for i in range(len(pca.explained_variance_ratio_)):
#         accum_epsilon = accum_epsilon + pca.explained_variance_ratio_[i]
#         if accum_epsilon > epsilon:
#             break
#     new_num_parameter = i # m dimensional
#     print "after PCA, " + str(new_num_parameter) + " parameters"
# 
#     print pca.components_[0:new_num_parameter, ]
# 
#     #x_i = pca.components_[i,] bin_data[]
#     bin_data_pca = []
#     for j in xrange(len(bin_data)):
#         x_j = []
#         for i in xrange(len(pca.components_[0:new_num_parameter, ])):
#             x_j_i = np.dot(pca.components_[i], bin_data[j])
#             x_j.append(x_j_i)
#         bin_data_pca.append(x_j)
#     bin_data_pca = np.array(bin_data_pca)
#     print bin_data_pca

    # for x in bin_data_pca:
    #     print list(x)
    # end of PCA

if __name__ == "__main__":
    main()