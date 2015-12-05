<<<<<<< HEAD
from __future__ import division
import json
import random
import itertools
import numpy as np
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
    
def main():
    bin_data, lbl_data = processfile('train.json')
    indices = shuffledata(len(bin_data), 10)

    # Added
    print "Starts doing PCA"
    num_parameter = len(bin_data[0])
    pca = PCA(n_components = 2)
    pca.fit(bin_data)
    print(pca.explained_variance_ratio_)
    
if __name__ == "__main__":
=======
from __future__ import division
import json
import random
import itertools
import numpy as np
from collections import Counter

def processfile(filename):
    with open(filename) as datafile:    
        data = json.load(datafile)
        alltags = itertools.chain(*[x["ingredients"] for x in data])
        lbl_data = [x["cuisine"] for x in data]
        tagset_c = Counter(alltags).most_common()
        tagset = [x[0] for x in tagset_c]
        bin_data = np.array([np.array([1 if t in tagset else 0 for t in x["ingredients"]]) for x in data])
        return bin_data, lbl_data
    
def shuffledata(datalen, numpart):
    idx = range(0, datalen)
    random.shuffle(idx)
    step = int(datalen/numpart)
    parts = []
    for i in range(0,numpart):
        parts.append(idx[i*step:(i+1)*step])
    return parts
    
def main():
    bin_data, lbl_data = processfile('train.json')
    indices = shuffledata(len(bin_data), 10)
    
if __name__ == "__main__":
>>>>>>> bee87acee23347dce9e6f71acd7485f0ef5435dd
    main()