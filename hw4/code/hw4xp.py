#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from distutils.version import LooseVersion, StrictVersion
import numpy as np
import random
import warnings
np.set_printoptions(precision=3, suppress=False, threshold=100000, linewidth=100000, edgeitems=10000)
np.random.seed(5)
import scipy.stats
import sklearn.cross_decomposition
import sklearn.linear_model  
import os
from sklearn.metrics.pairwise import cosine_similarity

print('The NumPy version is {0}.'.format(np.version.version))
print('The scikit-learn version is {0}.'.format(sklearn.__version__))
print('The SciPy version is {0}\n'.format(scipy.version.full_version)) # requires SciPy >= 0.16.0

if LooseVersion(scipy.version.full_version) < '0.16.0': 
    raise ValueError('SciPy version should be >= 0.16.0')

def load_word2vec_vectors(data_folder, words_to_extract):
    '''
    Load word vectors in memory
    The EMNLP 2015 paper uses GoogleNews-vectors-negative300.zip
    Can be downloaded on https://code.google.com/p/word2vec/
    '''
    extracted_word2vec_filename = os.path.join(data_folder, 'wordnet-synonyms+-out.txt')
        
    # Load GoogleNews-vectors-negative300: 
    # 1 row = 1 word + \t + its word vector (each float is separated by a comma)
    words2vec = {}
    vectors = np.loadtxt(extracted_word2vec_filename, dtype=str, delimiter='\t')
    for vector in vectors:
        words2vec[vector[0]] = map(float, vector[1].split(','))
    return words2vec

def convert_quantifier_to_float(quantifier):
    '''
    A quantifier is a string, we convert it to a float
    '''
    map_annotation_numerical = {'ALL': 1, 'MOST': 0.95, 'SOME': 0.35, 'FEW': 0.05, 'NO': 0, 'CONCEPT': 0} # EMNLP 2015 paper, page 5, paragraph 2
    return map_annotation_numerical[quantifier.upper()] 

def convert_float_to_quantifier(my_float):
    '''
    Convert a float to its quantifier equivalent (string)
    '''
    map_numerical_quantifier = {1:'ALL', 0.95:'MOST', 0.35:'SOME', 0.05: 'FEW', 0:'NO'} # EMNLP 2015 paper, page 5, paragraph 2
    return map_numerical_quantifier[my_float] 

def load_semantic_space_data(data_folder, dataset_filename):
    '''
    Load the model-theoretic space, a.k.a. semantic space
    '''
    mcrae_quantified_majority_data_filepath = os.path.join(data_folder, dataset_filename)
    semantic_space_data = np.loadtxt(mcrae_quantified_majority_data_filepath, dtype=str)
    
    # Spearman's rank correlation coefficient
    # Reproduces: "Using the numerical data, we can calculate the mean 
    #             Spearman rank correlation between the three annotators,
    #             which comes to 0.63."
    print('Spearman correlation between annotators:')
    for annotator_1, annotator_2 in [(3,4), (4,5), (3,5)]:
        spearman = scipy.stats.spearmanr(map(convert_quantifier_to_float, semantic_space_data[:, annotator_1]), 
                                         map(convert_quantifier_to_float, semantic_space_data[:, annotator_2]))
        print('spearman: {0}'.format(spearman))


    # Create model-theoretic space
    concepts = np.unique(semantic_space_data[:, 1])
    number_of_concepts = len(concepts)
    features = np.unique(semantic_space_data[:, 2])
    number_of_features = len(features)
    print('\nnumber_of_concepts: {0}; number_of_features: {1}; number_of_samples: {2}'.
          format(number_of_concepts, number_of_features, semantic_space_data.shape[0]))
    semantic_space = scipy.sparse.lil_matrix((number_of_concepts, number_of_features), dtype=float) 
    semantic_space_defined = scipy.sparse.lil_matrix((number_of_concepts, number_of_features), dtype=np.int) # indicate whether a point the semantic space was defined in the data set
    print('semantic_space.shape: {0}'.format(semantic_space.shape))
    
    # Populate model-theoretic space
    for semantic_space_sample in semantic_space_data:
        # Read sample
        concept = semantic_space_sample[1]
        feature = semantic_space_sample[2]
        quantifier = []
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[3]))
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[4]))
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[5]))
        
        # Put sample in semantic space
        concept_index = np.where(concepts==concept)
        feature_index = np.where(features==feature)
        mean_quantifier = np.mean(quantifier)
        if mean_quantifier > 0: 
            semantic_space[concept_index, feature_index] = mean_quantifier 
        semantic_space_defined[concept_index, feature_index] = 1
        
    return semantic_space, semantic_space_defined, concepts, features


def assess_prediction_quality(y_true, y_pred):
    print('scipy.stats.describe(y_true_all): {0}'.format(scipy.stats.describe(y_true)))
    print('scipy.stats.describe(y_pred_all): {0}'.format(scipy.stats.describe(y_pred)))        
    spearman = scipy.stats.spearmanr(y_true, y_pred)
    print('spearman: {0}'.format(spearman))
    return spearman



def load_semantic_space(data_folder, semantic_space_data_set):
    if semantic_space_data_set == 'qmr':
        semantic_space, semantic_space_defined, concepts, features = load_semantic_space_data(data_folder, 'mcrae-quantified-majority_no_header.txt')
        training_set_size = 400 # Following the EMNLP 2015 paper train/test split percentage (Table 2)
    elif semantic_space_data_set == 'ad':
        semantic_space, semantic_space_defined, concepts, features = load_semantic_space_data(data_folder, 'herbelot_iwcs13_data_standard_format.txt')
        training_set_size = 60 # Following the EMNLP 2015 paper train/test split percentage (Table 2)
    else:
        raise ValueError('Invalid semantic_space_data_set')

    return semantic_space, semantic_space_defined, concepts, features, training_set_size
    
def get_mode(data_folder,semantic_space_data_set, features, tidx):
    if semantic_space_data_set == 'qmr':
        modey = get_mode_helper(data_folder, 'mcrae-quantified-majority_no_header.txt', features, tidx)        
    elif semantic_space_data_set == 'ad':
        modey = get_mode_helper(data_folder, 'herbelot_iwcs13_data_standard_format.txt', features, tidx)    
    return modey

def get_mode_helper(data_folder, dataset_filename, features, tidx):
    '''
    Load the model-theoretic space, a.k.a. semantic space
    '''
    mcrae_quantified_majority_data_filepath = os.path.join(data_folder, dataset_filename)
    semantic_space_data = np.loadtxt(mcrae_quantified_majority_data_filepath, dtype=str)
    

    modey = np.zeros(len(features), dtype='float')
    
    for i in xrange(len(features)):
        search_me = semantic_space_data[ np.all( np.vstack(( semantic_space_data[:,2]==features[i], np.in1d(semantic_space_data[:,1], concepts[tidx]))), axis=0)  ][:,3:6]
        moderesult = scipy.stats.mode(search_me, axis=None)
        if (len(moderesult)==0 or len(moderesult[0]) ==0) :
            themode = 'NO'
        else:
            themode = moderesult[0][0]
        modey[i] = convert_quantifier_to_float(themode)
    
    
    return modey
    
    


def load_word_vectors(word_vector_type, data_folder, concepts):
    if word_vector_type == 'word2vec':
        words2vec = load_word2vec_vectors(data_folder, concepts)
    elif word_vector_type == 'random':
        words2vec = load_word2vec_vectors(data_folder, concepts)
        for key in words2vec.keys():
            words2vec[key] = np.random.rand(300).tolist()
    else:
        raise ValueError('Invalid word_vector_type')
    
    return words2vec
    
    
#def main():
## Set experiment parameters

# qmrad
#semantic_space_data_set = 'qmr'  # 541 concepts, 2201 features, 11.4 annotations
semantic_space_data_set = 'ad'  # 73 concepts, 54 features, 

#word_vector_type = 'random' #TODO
word_vector_type = 'word2vec'

#model_type = 'nearest-neighbor-in-train-set' #TODO
#model_type = 'mode' #TODO
#model_type = 'LR'
model_type = 'PLSR'

# Misc parameters
data_folder = os.path.join('..', 'data')

## Load data
semantic_space, semantic_space_defined, concepts, features, training_set_size = load_semantic_space(data_folder, semantic_space_data_set)
word_embeddings = load_word_vectors(word_vector_type, data_folder, concepts)


## PLS-regression
#Prepare data        
X = []
y_true = []
y_true_defined = []
for concept_id, concept in enumerate(concepts):
    if concept in word_embeddings:
        X.append(word_embeddings[concept])
        y_true.append(semantic_space[concept_id, :].todense().tolist()[0])
        y_true_defined.append(semantic_space_defined[concept_id, :].todense().tolist()[0])            

X = np.array(X)
y_true = np.array(y_true)
y_true_defined = np.array(y_true_defined)

# Perform several runs so that we get some statistically robust results
run_number = 0
total_run_number = 10
results = {}
results['spearmans'] = {}
results['spearmans']['only_annotated'] = {}
results['spearmans']['only_annotated']['SpearmanrResult'] = []
results['spearmans']['only_annotated']['correlation'] = []
results['train_set_idx'] = []
results['test_set_idx'] = []



while run_number < total_run_number:

    print('\n\n##### Run {0} out of {1} #####'.format(run_number+1, total_run_number))
    # Instantiate the model
    if model_type == 'PLSR':
        pls2 = sklearn.cross_decomposition.PLSRegression(n_components=20, max_iter=5000, scale=False, tol=1e-06)
    elif model_type == 'LR':
        pls2 = sklearn.linear_model.LinearRegression()        
    elif model_type in ['average', 'mode', 'nearest-neighbor-in-train-set']:
        pass
    else:
        raise ValueError('model_type is invalid')
    
    # Split train/test
    train_set_idx = np.array(random.sample(range(X.shape[0]), training_set_size))
    test_set_idx =  np.setdiff1d(range(0,X.shape[0]), train_set_idx)
    assert(np.intersect1d(train_set_idx, test_set_idx).shape[0]==0) # Ensure that the test set is not contaminated with training samples
    print('Data set size: {2}; train_set_idx.shape[0]: {0}; test_set_idx.shape[0]: {1}'.format(train_set_idx.shape[0], test_set_idx.shape[0], X.shape[0]))
    
    # Learn the model
    if model_type in ['LR', 'PLSR']:
        xx = X[train_set_idx, :]
        yy = y_true[train_set_idx, :]                        
        # Try-except to bypass sklearn's bug: https://github.com/scikit-learn/scikit-learn/issues/2089#issuecomment-152753095
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pls2.fit(xx, yy)
        except:
            print('sklearn.cross_decomposition.PLSRegression() crashed due to scikit-learn\'s bug https://github.com/scikit-learn/scikit-learn/issues/2089#issuecomment-152753095. Re-doing the run with a new train/test split.')
            continue
        
    elif model_type in ['mode']:
        modey = get_mode(data_folder, semantic_space_data_set, features, train_set_idx)
        
        
    elif model_type in ['nearest-neighbor-in-train-set']:
        xx = X[train_set_idx, :]
        yy = y_true[train_set_idx, :]                        
        

    # Use the learnt model to make predictions
    if model_type in ['LR', 'PLSR']:        
        y_pred = pls2.predict(X[test_set_idx, :])
    elif model_type in ['mode']:
        y_pred = np.array([modey]* len(test_set_idx) )
        # TODO: fill here
    elif model_type in ['nearest-neighbor-in-train-set']:
        xt = X[test_set_idx,:]
        cosinematrix = cosine_similarity(xt,xx)
        NNindex = np.argmax(cosinematrix, axis=1)
        y_pred = yy[NNindex,:]
        
        # TODO: fill here
        
    # Prepare test set
    y_test_true = y_true[test_set_idx, :]
    y_test_true_defined = y_true_defined[test_set_idx, :]
    
    y_pred_all = y_pred.flatten()
    y_test_true_all = y_test_true.flatten()
    y_test_true_defined_all = y_test_true_defined.flatten()
    
    # Assess model removing the features that are not specified in the human annotations, for each concept
    # (a concept has on average 11 annotated features in QMR)
    y_pred_all_only_annotated = []
    y_true_all_only_annotated = []
    for y_t, y_p, y_t_d in zip(y_test_true_all, y_pred_all, y_test_true_defined_all):
        if y_t_d == 1:
            y_pred_all_only_annotated.append(y_p)
            y_true_all_only_annotated.append(y_t)
    spearman = assess_prediction_quality(y_true_all_only_annotated, y_pred_all_only_annotated)
    
    # Save results
    results['test_set_idx'].append(test_set_idx)
    results['train_set_idx'].append(train_set_idx)
    results['spearmans']['only_annotated']['SpearmanrResult'].append(spearman)
    results['spearmans']['only_annotated']['correlation'].append(spearman.correlation)        

    run_number += 1
    
## Display results
print('\n\n\n##### Result summary for the {0} runs #####'.format(total_run_number))
for run_number in range(len(results['train_set_idx'])):
    print('Run {3:02d}: \tspearman.correlation: {0}\t train_set_idx: {1}\t test_set_idx: {2}'.format(results['spearmans']['only_annotated']['correlation'][run_number],
                                             results['train_set_idx'][run_number],
                                             results['test_set_idx'][run_number],
                                             run_number+1))

print('\nCorrelation stats over {1} runs (below are the only numbers you should report in the homework):\n{0}'.format(scipy.stats.describe(results['spearmans']['only_annotated']['correlation']), total_run_number))
    
#if __name__ == "__main__":
    #main()
    #cProfile.run('main()') # if you want to do some profiling