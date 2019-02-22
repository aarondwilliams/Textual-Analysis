# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 15:57:26 2018

@author: Aaron
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