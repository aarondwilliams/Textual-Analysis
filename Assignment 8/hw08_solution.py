# -*- coding: utf-8 -*-
"""

Assignment 08: Viterbi

@author: Aaron, Ashwin, Niral

"""
import numpy as np

def infer_states(obs,pi,A,B):
    
    num_states = len(pi)
    num_obs = len(obs)
    
    prob_array = np.zeros((num_states,num_obs))
    prob_array[:,0] = np.multiply(pi,B[:,obs[0]])
    state_array = np.zeros((num_states,num_obs))
    state_array[:,0] = range(num_states)
    
    for T in range(1,num_obs):
        
        for j in range(num_states):
            
            prev_state = int(state_array[j,T-1])
            curr_obs = obs[T]
            out_vals = np.multiply(A[prev_state,:],B[:,curr_obs])
            
            prob_array[j,T] = prob_array[j,T-1]*max(out_vals)
            state_array[j,T] = np.argmax(out_vals)
    
    bestpathprob = max(prob_array[:,num_obs-1])
    bestpath = state_array[np.argmax(prob_array[:,num_obs-1]),:]
    
    return list(bestpath),bestpathprob