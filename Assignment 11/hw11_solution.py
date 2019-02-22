# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:40:37 2018

HW11 - LDA
@author: Aaron, Ashwin, Niral
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

def lda_solve(documents):
    
    full_doc = [item for sublist in documents for item in sublist]
    Vocabulary = list(set(full_doc))
    Distribution = np.zeros((len(documents),len(Vocabulary)))
    
    which_list = 0
    for sublist in documents:
        for word in sublist:
            Distribution[which_list,Vocabulary.index(word)] += 1
        which_list += 1
    
    lda = LatentDirichletAllocation(n_components = 3, random_state = 0)
    lda.fit(Distribution)
    
    return lda.components_,Vocabulary
    
    
def lda(vocabulary,phi,theta,N):
    
    w = []
    
    for doc in theta:
        w_sub = []
        
        by_topic = np.multiply(doc,np.transpose(phi))
        vocab_prob = np.sum(by_topic, axis=1)
        
        indices = np.random.choice(len(vocabulary),N,p=vocab_prob)
        w_sub = [vocabulary[i] for i in indices]

        w.append(w_sub)
                    
    return w