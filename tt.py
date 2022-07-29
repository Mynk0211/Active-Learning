#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:13:21 2022

@author: clara
"""

import pickle


def save_count(bm_algo):
    o = open(f"count/{bm_algo}.pkl", "rb")
    c = pickle.load(o)
    o.close()
    return c
    




    
print(save_count("mvdr"))
   
print(save_count("gcf"))
   
print(save_count("das"))
   
print(save_count("dmas"))

