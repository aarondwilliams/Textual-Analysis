# -*- coding: utf-8 -*-
"""
ECE 590: Natural Language Processing
Created on Mon Sep 24 08:55:04 2018

@author: Aaron, Niral, Ashwin
"""

def levenshtein_dp(a,b):
    
    len_a = len(a)+1
    len_b = len(b)+1
    
    array = [[0] * (len_b) for i in range(len_a)]
    
    for i in range(len_a):
        array[i][0] = i
    
    for i in range(len_b):
        array[0][i] = i

    for i in range(1,len_a):
        for j in range(1,len_b):           
            del_cost = array[i-1][j] + 1
            ins_cost = array[i][j-1] + 1
            sub_cost = array[i-1][j-1]
            if a[i-1] != b[j-1]:
                sub_cost += 1
            array[i][j] = min(del_cost,ins_cost,sub_cost)
    
    return array[len_a-1][len_b-1]

def suggest(word,dictionary):
    
    min_val = len(word)
    word_list = []
    
    for dicts in dictionary:
        curr_val = levenshtein_dp(word,dicts)
        
        if curr_val < min_val:
            word_list = [dicts]
            min_val = curr_val
        elif curr_val == min_val:
            word_list.append(dicts)
            
    return word_list