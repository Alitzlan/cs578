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

def svdfit(train_data, train_label, ncomponents=1000):
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
    elif fitfunc is not None and data is not None:
        print "file not exist. training model..."
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

def logisticfit(train_data, train_label, regulation=0.00001):
    classifier = linear_model.LogisticRegression(C=1/regulation)
    classifier.fit(train_data, train_label)
    return classifier

def get_accuracy(expected, predicted):
    accuracy = 1 - (expected != predicted).sum() / len(expected)
    print ("accuracy: %f" % (accuracy))
    return accuracy

def gnb_fit(train_data, train_lbl_data):
    from sklearn.naive_bayes import GaussianNB
    print "Starts gnb"

    gnb = GaussianNB()

    gnb.fit(train_data, train_lbl_data)
    return gnb

def bnb_fit(train_data, train_lbl_data):
    from sklearn.naive_bayes import BernoulliNB
    print "Starts bnb"

    bnb = BernoulliNB()
    bnb.fit(train_data, train_lbl_data)
    return bnb

def mnb_fit(train_data, train_lbl_data):
    from sklearn.naive_bayes import MultinomialNB
    print "Starts mnb"

    mnb = MultinomialNB()

    mnb.fit(train_data, train_lbl_data)
    return mnb

def dt_fit(train_data, train_lbl_data):
    from sklearn import tree
    print "Starts DT"
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_lbl_data)
    return clf

def k_fold(bin_data, lbl_data, indices, fit_func):
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

        model = fit_func(train_data,train_lbl_data)
        
        print ("--------------- iteration %d ----------------" %(i))
        predicted = model.predict(test_data)
        get_accuracy (test_lbl_data, predicted)
        #print ("Number of mislabeled points out of a total %d points : %d" % (test_data.shape[0], (test_lbl_data != predicted).sum()))


def loadpca(bin_data, filename):
    objfilename = filename + '.dat'
    num_parameter = len(bin_data[0])
    print "Originally " + str(num_parameter) + " parameters"
    if os.path.isfile(objfilename):
        with open(objfilename,'rb') as objfile:
            bin_data_pca = np.load(objfile)
    else:
        pca = PCA(n_components=num_parameter)
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

        #x_i = pca.components_[i,] bin_data[]
        bin_data_pca = []
        for j in xrange(len(bin_data)):
            x_j = []
            for i in xrange(len(pca.components_[0:new_num_parameter, ])):
                x_j_i = np.dot(pca.components_[i], bin_data[j])
                print x_j_i
                x_j.append(x_j_i)
            bin_data_pca.append(x_j)
        bin_data_pca = np.array(bin_data_pca)

        with open(objfilename,'wb') as objfile:
            np.save(objfile, bin_data_pca)
    
    return bin_data_pca


def pca(bin_data):
    
    # Added for PCA
    print "Starts doing PCA"
    
    filename = 'train.json.pca'
    bin_data_pca = loadpca(bin_data, filename)

    bin_data_pca = np.array(bin_data_pca, dtype="float")
    
    print bin_data_pca
    return bin_data_pca

def main():
    bin_data, lbl_data, tag_set = loaddata('train.json')
    tbin_data, tlbl_data = subset(bin_data, lbl_data, start=5000, end=6000)
    bin_data, lbl_data = subset(bin_data, lbl_data, end=5000)
    indices = shuffledata(len(bin_data), 10)
    # restore format
    bin_data = np.array(bin_data,dtype="float")
    print 'finish processing data'

    svd = loadmodel("svd", svdfit, bin_data, None)
    bin_data = svd.transform(bin_data)
    nn = loadmodel("neuralmodel", neuralfit, bin_data, lbl_data)
    res = nn.predict(svd.transform(tbin_data))
    print res
    lr = loadmodel("logisticmodel", logisticfit, bin_data, lbl_data)
    print("Logistic regression using raw features:\n%s\n" % (
    metrics.classification_report(
        tlbl_data,
        lr.predict(svd.transform(tbin_data)))))

if __name__ == "__main__":
    main()