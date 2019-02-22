# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 08:44:46 2018

@author: Aaron
"""

import nltk 
import numpy as np
from nltk.corpus import brown

def get_word_vector():
    
    review_corpus = brown.words(categories='reviews')
    editorial_corpus = brown.words(categories='editorial')
    news_corpus = brown.words(categories='news')
    total_corpus = brown.words(categories=['reviews','editorial','news'])
    
    review_map = {word: 1 for word in review_corpus}
    editorial_map = {word: 1 for word in editorial_corpus}
    news_map = {word: 1 for word in news_corpus}
    total_map = {word: 1 for word in total_corpus}
    
    review_counts = np.zeros(len(review_map))
    editorial_counts = np.zeros(len(editorial_map))
    news_counts = np.zeros(len(news_map))
    total_dfs = np.zeros(len(total_map))
    
    for word in total_corpus:
        if word in review_map:
            total_dfs[total_map[word]] += 1
        if word in editorial_map:
            total_dfs[total_map[word]] += 1
        if word in news_map:
            total_dfs[total_map[word]] += 1
    total_dfs = np.log10(3/total_dfs)
    
    for word in review_corpus:
        review_counts[review_map[word]] += 1
    for word in editorial_corpus:
        editorial_counts[editorial_map[word]] += 1
    for word in news_corpus:
        news_counts[news_map[word]] += 1
        
    final_vector = np.zeros((len(total_map),3))
    for word in total_map:
        if word in review_map:
            final_vector[0,total_map[word]] = np.log10(1 + review_counts[review_map[word]]/len(review_counts))*total_dfs[total_map[word]]
        if word in editorial_map:
            final_vector[0,total_map[word]] = np.log10(1 + editorial_counts[editorial_map[word]]/len(editorial_counts))*total_dfs[total_map[word]]
        if word in news_map:
            final_vector[0,total_map[word]] = np.log10(1 + news_counts[news_map[word]]/len(news_counts))*total_dfs[total_map[word]]

    return final_vector, total_map

def related_words(word):
    
    n = 10
    
    matrix, word_map = get_word_vector()
    print(word_map)
    des_vec = matrix[word_map[word]]
    most_related = np.zeros(len(word_map))
    
    for words in word_map:
        most_related[word_map[word]] = np.dot(matrix[word_map[word]],des_vec)
    
    best_indexes = np.argpartition(most_related, n+1)[(n+1):]
    
    print(len(word_map))
    best_words = []
    for i in best_indexes:
        best_words.append(word_map[i])
    best_words.remove(word)
            
    return best_words