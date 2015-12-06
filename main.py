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
    main()