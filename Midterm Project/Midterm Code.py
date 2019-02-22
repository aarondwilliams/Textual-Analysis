# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:04:32 2018

Midterm Project.  Training, evaluating, and testing

@author: Aaron
"""

import nltk 
import numpy as np
from nltk.corpus import brown

UNIQUE_WORDS = [""]
NUM_WORDS = 0
UNIQUE_TAGS = [""]
NUM_TAGS = 0

#nltk.download('brown')
#nltk.download('universal_tagset')

def get_word(index):
    if index != NUM_WORDS:
        return UNIQUE_WORDS[index]
    else:
        return "unkown"

def get_tag(index):
    return UNIQUE_TAGS[index]

def list_to_int(in_set,in_map):
    '''Converts a list of strings to a 1d array of mapped unsigned integers'''
    len_set = len(in_set)
    out_set = np.zeros(len_set).astype(int)
    for i in range(len_set):
        if in_set[i] in in_map:
            out_set[i] = in_map.index(in_set[i])
        else:
            '''If value is not in map, maps to the length of the map'''
            out_set[i] = len(in_map)
        
        #if i%10000 == 0:
            #print(str(i) + "/" + str(len_set))
    
    return out_set

def training(corpus_words,corpus_tags):
    
    global UNIQUE_WORDS
    global NUM_WORDS
    global UNIQUE_TAGS
    global NUM_TAGS
    UNIQUE_WORDS = list(set(corpus_words))
    NUM_WORDS = len(UNIQUE_WORDS)
    UNIQUE_TAGS = list(set(corpus_tags))
    NUM_TAGS = len(UNIQUE_TAGS)
    
    corpus_int_words = list_to_int(corpus_words,UNIQUE_WORDS)
    corpus_int_tags = list_to_int(corpus_tags,UNIQUE_TAGS)
        
    len_corp = len(corpus_int_words)
    
    wprobs = np.zeros((NUM_WORDS,NUM_TAGS))
    '''tprobs is our transition matrix, length 2 for forward and back'''
    tprobs = np.zeros((2,NUM_WORDS,NUM_TAGS,NUM_TAGS))
    
    for i in range(len_corp):
        '''Gets frequency counts'''
        corp_word = corpus_int_words[i]
        corp_tag = corpus_int_tags[i]
        wprobs[corp_word,corp_tag] += 1
        
        if i != 0 :
            tprobs[0,corp_word,corp_tag,corpus_int_tags[i-1]] += 1
        if i != len_corp-1:
            tprobs[1,corp_word,corp_tag,corpus_int_tags[i+1]] += 1

    return wprobs, tprobs
    
def seperate_training(wprobs,tprobs):
    '''Finalizes probability matrices for two algorithms'''
    wprobs1 = wprobs + 1
    tprobs1 = tprobs[0] + 1
    tsquash = np.sum(tprobs, axis=1)
    tsquash1 = tsquash[0] + 1
    
    for i in range(NUM_WORDS):
        wprobs[i] = wprobs[i]/np.sum(wprobs[i])
        wprobs1[i] = np.log(wprobs1[i]/np.sum(wprobs1[i]))
        tprobs1[i] = np.log(tprobs1[i]/np.sum(tprobs1[i]))
        for j in [0,1]:
            tprobs[j,i] = tprobs[j,i]/(np.sum(tprobs[j,i])+.01)
    
    for i in range(NUM_TAGS):
        tsquash1[:,i] = np.log(tsquash1[:,i]/np.sum(tsquash1[:,i]))
        for j in [0,1]:
            tsquash[j,:,i] = tsquash[j,:,i]/(np.sum(tsquash[j,:,i])+.01)
    
    return wprobs, wprobs1, tprobs, tprobs1, tsquash, tsquash1
    
def fwd_viterbi(words,tags,wprobs,tprobs,tsquash):
    
    len_corp = len(words)
    prob_array = np.zeros((len_corp,NUM_TAGS))
    if words[0] != NUM_WORDS:
        prob_array[0] = wprobs[words[0],range(NUM_TAGS)]
    state_array = np.zeros((len_corp,NUM_TAGS)).astype(int)
    state_array[0] = range(NUM_TAGS)
    for T in range(1,len_corp):
        
        curr_word = words[T]
        prev_word = words[T-1]
        if curr_word == NUM_WORDS:
            '''Deals with words that weren't found in training'''
            prob_array[T] = prob_array[T-1] + np.amax(tsquash[:,state_array[T-1]], axis = 0)
            state_array[T] = np.argmax(tsquash[:,state_array[T-1]], axis = 0)
        else:
            '''Applies forward word transition probabilities and word state probabilities'''
            trans_probs = tprobs[curr_word,:,state_array[T-1]]
            for i in range(NUM_TAGS):
                trans_probs[:,i] += wprobs[curr_word,:]
            prob_array[T] = prob_array[T-1] + np.amax(trans_probs, axis = 0)
            state_array[T] = np.argmax(trans_probs, axis = 0)

    best_prob = np.argmax(prob_array[len_corp-1])
    best_path = state_array[:,best_prob]
    
    return sum(best_path == tags)/len_corp, best_path

def get_array(subset,first_tag,last_tag,wprobs,tprobs,tsquash):
    
    len_sub = len(subset)
    start_states = np.ones(len_sub).astype(int)*NUM_TAGS
    start_states[0] = first_tag
    start_states[len_sub-1] = last_tag
    
    fwd_prob = 0.
    fwd_states = start_states
    if first_tag != NUM_TAGS:
        '''Forward propogation if forward anchored'''
        fwd_prob = 1.
        end = len_sub-1
        if last_tag == NUM_TAGS:
            #For last term in set if applicable
            end += 1
        for T in range(1,end):
            
            curr_prob = np.ones(NUM_TAGS)
            '''Factor in transition from previous state'''
            if subset[T-1] != NUM_WORDS:
                curr_prob *= tprobs[1,subset[T-1],fwd_states[T-1],:]
            else:
                curr_prob *= tsquash[1,fwd_states[T-1],:]
            
            '''Transition to current state'''
            if subset[T] != NUM_WORDS:
                curr_prob *= tprobs[0,subset[T],:,fwd_states[T-1]]
                curr_prob *= wprobs[subset[T],:]
            else:
                curr_prob *= tsquash[0,:,fwd_states[T-1]]
            
            fwd_prob *= max(curr_prob)
            fwd_states[T] = np.argmax(curr_prob)
            
        '''Final previous state transition to final anchor'''
        if last_tag != NUM_TAGS:
            fwd_prob *= tprobs[0,subset[len_sub-1],fwd_states[len_sub-1],fwd_states[len_sub-2]]
        else:
            fwd_prob *= tsquash[0,fwd_states[len_sub-1],fwd_states[len_sub-2]]
        
    bkwd_prob = 0.
    bkwd_states = start_states
    if last_tag != NUM_TAGS:
        '''Backward propogation if backward anchored'''
        bkwd_prob = 1.
        begin = 0
        if first_tag == NUM_TAGS:
            #For First term in set if applicable
            begin = -1        
        
        for T in range(len_sub-2,begin,-1):
            curr_prob = np.ones(NUM_TAGS)
            if subset[T+1] != NUM_WORDS:
                curr_prob *= tprobs[0,subset[T+1],bkwd_states[T+1],:]
            else:
                curr_prob *= tsquash[0,bkwd_states[T+1],:]
            
            if subset[T] != NUM_WORDS:
                curr_prob *= tprobs[1,subset[T],:,bkwd_states[T+1]]
                curr_prob *= wprobs[subset[T],:]
            else:
                curr_prob *= tsquash[1,:,bkwd_states[T+1]]
            
            bkwd_prob *= max(curr_prob)
            bkwd_states[T] = np.argmax(curr_prob)
        
        if first_tag != NUM_TAGS:
            bkwd_prob *= tprobs[1,subset[0],bkwd_states[0],bkwd_states[1]]
        else:
            bkwd_prob *= tsquash[1,bkwd_states[0],bkwd_states[1]]
    
    if fwd_prob >= bkwd_prob:
        return fwd_states
    return bkwd_states

def bi_dir_method(words,tags,wprobs,tprobs,tsquash):
    
    len_corp = len(words)
    state_array = np.zeros(len_corp).astype(int)
    for T in range(len_corp):
        '''Picks out "sure thing" probabilities'''
        if words[T] != NUM_WORDS:
            if max(wprobs[words[T],:]) >= 0.9:
                state_array[T] = np.argmax(wprobs[words[T],:])
            else:
                state_array[T] = NUM_TAGS
        else:
            state_array[T] = NUM_TAGS
    
    T = 0
    while T < len_corp:
        first_T = T
        if state_array[T] == NUM_TAGS:
            '''Picks out subset of data'''
            while state_array[T] == NUM_TAGS and T < len_corp:
                T += 1
            if first_T > 0:
                first_T -= 1
            if T < len_corp:
                T += 1
            
            state_array[first_T:T] = get_array(words[first_T:T],state_array[first_T],state_array[T-1],wprobs,tprobs,tsquash)
        else:
            T += 1  
    
    return sum(state_array == tags)/len_corp, state_array

def primary():

    print("Training on the adventure category ...")
    train_corpus = brown.tagged_words(categories='adventure', tagset='universal')
    
    wprobs, tprobs = training([y[0] for y in train_corpus],[y[1] for y in train_corpus])
    print("Corpus Trained")
    
    wprobs, wprobs1, tprobs, tprobs1, tsquash, tsquash1 = seperate_training(wprobs,tprobs)
    
    test_cats = brown.categories()
    vit_matrix = np.zeros((NUM_TAGS,NUM_TAGS))
    bi_matrix = np.zeros((NUM_TAGS,NUM_TAGS))
    for test_cat in test_cats:
        
        print("Testing on " + test_cat + "...")
        test_corpus = brown.tagged_words(categories=test_cat, tagset='universal')
        test_words = list_to_int([y[0] for y in test_corpus],UNIQUE_WORDS)
        test_tags = list_to_int([y[1] for y in test_corpus],UNIQUE_TAGS)
        
        fwd_vit_acc, vit_set = fwd_viterbi(test_words,test_tags,wprobs1,tprobs1,tsquash1)
        print("Forward Viterbi resulted in accuracy: " + str(fwd_vit_acc))
        bi_dir_acc, bi_set = bi_dir_method(test_words,test_tags,wprobs,tprobs,tsquash)
        print("Bidirection Method resulted in accuracy: " + str(bi_dir_acc))
        
        for i in range(len(test_tags)):
            vit_matrix[test_tags[i],vit_set[i]] += 1
            bi_matrix[test_tags[i],bi_set[i]] += 1
            
    np.set_printoptions(suppress=True)
    print("The Confusion Matrix for Viterbi:")
    print(vit_matrix)
    print("The Confusion Matrix for Bidirectional:")
    print(bi_matrix)
    for i in range(NUM_TAGS):
        print("Accuracy of '" +  UNIQUE_TAGS[i] + "' tagging on Viterbi: " + str(vit_matrix[i,i]/sum(vit_matrix[i])))
        print("Accuracy of '" +  UNIQUE_TAGS[i] + "' tagging on Bidirection: " + str(bi_matrix[i,i]/sum(bi_matrix[i])))

        print("misguess of '" +  UNIQUE_TAGS[i] + "' on Viterbi: " + str(1 - vit_matrix[i,i]/sum(vit_matrix[:,i])))
        print("misguess of '" +  UNIQUE_TAGS[i] + "' on Bidirection: " + str(1 - bi_matrix[i,i]/sum(bi_matrix[:,i])))

        
if __name__ == "__main__":
    primary()