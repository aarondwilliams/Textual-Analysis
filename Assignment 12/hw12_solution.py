# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 08:36:56 2018

Assignment 12: Lexicon Expansion
@author: Aaron, Ashwin, Niral
"""

from nltk.corpus import wordnet as wn

def score_document(document, positive_seeds, negative_seeds):

    score = 0
    curr_pos_list = [p for p in positive_seeds]
    curr_neg_list = [n for n in negative_seeds]
    
    for i in range(1):
        new_seeds = []
        for seed in curr_pos_list:
            ss = wn.synsets(seed)
            for syn in ss:
                for lemma in syn.lemmas():
                    if lemma.name() not in curr_pos_list:
                        new_seeds.append(lemma.name())
        
        curr_pos_list += new_seeds
        
        new_seeds = []
        for seed in curr_neg_list:
            ss = wn.synsets(seed)
            for syn in ss:
                for lemma in syn.lemmas():
                    if lemma.name() not in curr_neg_list:
                        new_seeds.append(lemma.name())
    
        curr_neg_list += new_seeds
            
    for word in document:
        if word in curr_pos_list:
            score += 1
        elif word in curr_neg_list:
            score -= 1
    
    return score