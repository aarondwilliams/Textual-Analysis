# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:43:16 2018

@author: Aaron
"""

def finish_sentence(sentence,n,corpus):
        
    last_word = sentence[-1]
    c = 0
    while last_word != "." or last_word != "?" or last_word != "!":
        
        gram = sentence[(3-n+c):(2+c+1)]
        
        list_words = []
        for i in range (0,len(corpus)-n+1):
            
            condition = True
            for j in range(0,n):
                if gram[j] != corpus[i+j]:
                    condition = False
            
            if condition:
                list_words.append(corpus[i+n+1])
        
        print(list_words)
        last_word = max(list_words, key=list_words.get)
        sentence.append(last_word)
        c += 1
    
    return sentence