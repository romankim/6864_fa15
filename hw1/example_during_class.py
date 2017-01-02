
# coding: utf-8


from __future__ import print_function
from __future__ import division

import numpy as np
import languagemodel as lm


np.random.seed(1)  # for reproducibility

corpus_train = lm.readCorpus("data/train.txt")
corpus_dev   = lm.readCorpus("data/dev.txt")
corpus_test  = lm.readCorpus("data/test.txt")

# build a common index (words to integers), mapping rare words (less than 5 occurences) to index 0
# nwords = vocabulary size for the models that only see the indexes

w2index,nwords = lm.buildIndex(corpus_train+corpus_dev+corpus_test)

# find words that appear in the training set so we can deal with new words separately
count_train = np.zeros((nwords,))
for snt in corpus_train:
    for w in snt:
        count_train[w2index[w]] += 1

#%% Bigram configuration
# Bigram model as a baseline
alpha           = 0.02 # add-alpha smoothing
probB           = lm.bigramLM(corpus_train, w2index, nwords,alpha)
LLB, N          = 0.0, 0
bi              = lm.ngramGen(corpus_dev, w2index, 2)

#%% test the empirical bigram on dev.txt

for w in bi:
    if (count_train[w[1]]>0): # for now, skip target words not seen in training, to avoid LLB being inf with alpha is zero
        LLB += np.log(probB[w[0], w[1]])
        N += 1
print("Bi-gram Dev LL = {0}".format(LLB / N))
# Train the probB on the train.txt, and test the normalized LL on the dev.txt

#%% test the NN-generated prob distribution on dev.txt

# Network model
print("\nNetwork model training:")
n        = 3    # Length of n-gram 
dim      = 10   # Word vector dimension
hdim     = 40  # Hidden units
neurallm = lm.neuralLM(dim, n, hdim, nwords)  # The network model

ngrams = lm.ngramGen(corpus_train,w2index,n)
ngrams2 = lm.ngramGen(corpus_dev,w2index,n)

lrate = 0.5  # Learning rate
for it in xrange(10): # passes through the training data
    LL, N  = 0.0, 0 # Average log-likelihood, number of ngrams    
    for ng in ngrams:
        pr = neurallm.update(ng,lrate)
        LL += np.log(pr)
        N  += 1
    print('Train:\t{0}\tLL = {1}'.format(it, LL / N)) 

    #Dev set
    LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
    for ng in ngrams2:
        if (count_train[ng[-1]]>0): # for now, skip target words not seen in training
            pr = neurallm.prob(ng)
            LL += np.log(pr)
            N  += 1
    print('Dev:\t{0}\tLL = {1}'.format(it, LL / N)) 





#%% test session 2


# Network model
print("\nNetwork model training:")

dimlist   = range(5,16)
hdimlist  = range(25,41)
paramlist = [ (dim,hdim) for dim in dimlist for hdim in hdimlist ]

returnStr = []

for (dim,hdim) in paramlist:    
    n        = 3    # Length of n-gram 
    #dim      = 10   # Word vector dimension
    #hdim     = 30  # Hidden units
    neurallm = lm.neuralLM(dim, n, hdim, nwords)  # The network model
    
    ngrams = lm.ngramGen(corpus_train,w2index,n)
    ngrams2 = lm.ngramGen(corpus_dev,w2index,n)
    
    lrate = 0.5  # Learning rate
    for it in xrange(10): # passes through the training data
        LL, N  = 0.0, 0 # Average log-likelihood, number of ngrams    
        for ng in ngrams:
            pr = neurallm.update(ng,lrate)
            LL += np.log(pr)
            N  += 1
        #if it == 9:
        #    print('Train:\t{0}\tLL = {1}\tn = {2:<4d} d = {3:<4d} m = {4:<4d}'.format(it, LL / N, n, dim, hdim)) 
        #Dev set
        LL, N = 0.0, 0 # Average log-likelihood, number of ngrams
        for ng in ngrams2:
            if (count_train[ng[-1]]>0): # for now, skip target words not seen in training
                pr = neurallm.prob(ng)
                LL += np.log(pr)
                N  += 1
        if it == 9:
            returnStr.append('Dev:\t{0}\tLL = {1}\tn = {2:<4d} d = {3:<4d} m = {4:<4d}'.format(it, LL / N, n, dim, hdim))                 

f = open('output_during_class.txt', 'w')
f.writelines(returnStr)
f.close()

