from __future__ import division
import json
import random
import itertools
import pickle
import numpy as np
import os.path
import sys
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
    
def shuffledata(datalen, numpart):
    idx = range(0, datalen)
    random.shuffle(idx)
    step = int(datalen/numpart)
    parts = []
    for i in range(0,numpart):
        parts.append(idx[i*step:(i+1)*step])
    return parts
    
def main():
    bin_data, lbl_data, tag_set = loaddata('train.json')
    indices = shuffledata(len(bin_data), 10)
    print 'finish processing data'
    
if __name__ == "__main__":
    main()