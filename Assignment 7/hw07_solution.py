# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 08:37:25 2018

Assignment 7: Textual Analysis
@author: Aaron, Niral, Ashwin
"""

import numpy as np

def transition_matrix(corpus):
    
    t_matrix = np.ones((27,27))
    
    for word in corpus:
        
        word_len = len(word)
        
        for i in range(word_len):
            
            cl = ord(word[i]) - ord('a') 
            
            if i == 0:
                t_matrix[26][cl] += 1
            
            if i+1 == word_len:
                t_matrix[cl][26] += 1
            else:
                nl = ord(word[i+1]) - ord('a')
                t_matrix[cl][nl] += 1
        
    for i in range(27):
        
        sum_l = sum(t_matrix[i])
        
        for j in range(27):
            t_matrix[i][j] = np.log(t_matrix[i][j]/sum_l)
    
    return t_matrix

def most_likely_word(corpus,matrix,n):
    
    word_list = []
    for word in corpus:
        if len(word) == n:
            word_list.append(word)
        
    best_word = "Did not find"
    best_prob = -100
    
    for word in word_list:
        
        curr_prob = 0
        for i in range(n):
            
            cl = ord(word[i]) - ord('a') 
            
            if i == 0:
                curr_prob += matrix[26][cl]
            if i+1 == n:
                curr_prob += matrix[cl][26]
            else:
                nl = ord(word[i+1]) - ord('a')
                curr_prob += matrix[cl][nl]
        
        if curr_prob > best_prob:
            best_word = word
            best_prob = curr_prob  
    
    return best_word