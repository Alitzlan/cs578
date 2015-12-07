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

def svdfit(train_data, ncomponents=1000):
    svd = TruncatedSVD(n_components=ncomponents)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    lsa.fit(train_data)
    return lsa

def loadmodel(filename=None, fitfunc=None, data=None, label=None):
    model = None
    if filename is not None and os.path.isfile(filename):
        with open(filename, "rb") as modelfile:
            print "loading model from file..."
            model = pickle.load(modelfile)
    elif fitfunc is not None and data is not None and label is not None:
        model = fitfunc(data, label)
        if filename is not None:
            with open(filename, "wb") as modelfile:
                print "saving model to file..."
                pickle.dump(model, modelfile)
    else:
        raise Exception("Load model failed", "Not enough parameter.")
    return model

def neuralfit(train_data, train_label, regulation=0.00001, learnrate=0.001, ncomponents=320, iteration=10):
    logistic = linear_model.LogisticRegression(C=1/regulation)
    rbm = BernoulliRBM(verbose=True, learning_rate=learnrate, n_components=ncomponents, n_iter=iteration)
    classifier = Pipeline(steps=[('rbm', rbm),('logistic', logistic)])
    classifier.fit(train_data, train_label)
    return classifier
    
def main():
    bin_data, lbl_data, tag_set = loaddata('train.json')
    tbin_data, tlbl_data = subset(bin_data, lbl_data, start=5000, end=6000)
    bin_data, lbl_data = subset(bin_data, lbl_data, end=5000)
    indices = shuffledata(len(bin_data), 10)
    # restore format
    bin_data = np.array(bin_data,dtype="float")
    print 'finish processing data'

    svd = svdfit(bin_data)
    bin_data = svd.transform(bin_data)
    nn = loadmodel("neuralmodel", neuralfit, bin_data, lbl_data)
    res = nn.predict(svd.transform(tbin_data))
    print res
                     
if __name__ == "__main__":
    main()