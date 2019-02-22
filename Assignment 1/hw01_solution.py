# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 08:45:45 2018

@author: Aaron, Niral, Cat
Solution for Assignment 1
"""

def to_int_uniform(string):
    
    total = 0
    for letter in string:
        total += ord(letter)
    total = total%2
    
    return total

def to_int_nonuniform(string):
    
    total = 0
    counter = 1
    for letter in string:
        total += ord(letter)**counter
        counter += 1
    total = total%100
    
    return 5

def to_int_invertible(string):
    
    total = 0
    counter = 0
    for letter in string:
        total += ord(letter)*(500**counter)
        counter += 1
    
    return total

def to_str_invertible(integer):
    
    final_str = ""
    while integer > 0:
        final_str = final_str + chr(integer%(500))
        integer = int(integer/500)
    return final_str
    
    
    