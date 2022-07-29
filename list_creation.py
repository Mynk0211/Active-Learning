#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 23:59:25 2022

@author: mayankkatare
"""



from function import *
import os
import numpy as np
import pickle
from tqdm import tqdm



def get_rawdata(file_type = "data_cube"):
    ls = []
    for i  in tqdm(range(1,6)):
        path = f"Verasonics/Mayank/ActiveLearning/dataset_phantom_{i}_{i}/{file_type}_PW_{i}.mat"
        #path = f"web_data1/{file_type}_PW_{i}.mat"
        l = get_datacube(path).detach().numpy()
        ls.append(l)
    
    sample_list = ls
    file_name = "lists/data_cube.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()
    
  
    
        


def get_data(file_type):
    ls = []
    for i  in tqdm(range(1,6)):
        path = f"Verasonics/Mayank/ActiveLearning/dataset_phantom_{i}_{i}/{file_type}_PW_{i}.mat"
        #path = f"web_data1/{file_type}_PW_{i}.mat"
        l = get_image(path , file_type).detach().numpy()
        ls.append(l)
  
    sample_list = ls
    file_name = f"lists/{file_type}.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

get_rawdata()



get_data("DAS")
get_data("DMAS")
get_data("MVDR")
get_data("GCF")



'''
open_file = open(file_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

'''
