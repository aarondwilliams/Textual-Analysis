# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:40:25 2018

Homework 10 - Latent Dirichlet Allocation
@author: Aaron,Ashwin,Niral
"""

import numpy as np

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