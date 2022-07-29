#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:07:34 2022

@author: mayankkatare
"""

#importing libraries
import torch
import torchvision
import gc
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.io 
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.signal import hilbert
import random
import matplotlib.pyplot as plt 
import monai
from monai.networks.layers import HilbertTransform
import math
import pickle


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


## To read the datacube.
def get_datacube(path):
  data =  h5py.File(path,'r')  
  data = data.get('dataCube') ## for web data data_cube is the key and dataCube for new acquired data.
  input_data = np.array(data) 
  
  return torch.tensor(input_data , dtype=torch.float32).permute([0 ,2,1])


# To read the image files.
def get_image(path, name): 
  data =  h5py.File(path,'r')
  path = "beamformedData" + name +"Image"
  data = data.get(path)
  input_data = np.array(data)  
  
  return torch.tensor(input_data , dtype=torch.float32).permute([1,0])



def save_count(bm_algo):
    

    o = open(f"count/{bm_algo}.pkl", "rb")
    c = pickle.load(o)
    o.close()
    c = c+1
    file_name = f"count/{bm_algo}.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(c, open_file)
    open_file.close()
    
    



# To check errors in dataset
def check(image):
  
  t = image > 0
  t = t.bool().int()
  
  
  if torch.sum(t) > 0:

    image = image/20
    new_image = torch.pow(10.0,image)
    image = 20*torch.log10(new_image/torch.max(new_image))

  return image



def training(model , raw , p , optimizer , criterion):
    train_loss=0
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    input_Data = torch.tensor(raw , dtype=torch.float32)
    im = torch.tensor(p[1] , dtype=torch.float32)
    
    im = check(im)
    input_Data, output_data = input_Data.to(dev) , im.to(dev)
    input_Data = torch.unsqueeze(input_Data , 0)
    output_data = torch.unsqueeze(output_data,0)
   
    
    optimizer.zero_grad()
    weight= model(input_Data)
    beamformed = torch.mul(weight , input_Data)
    beamformed_sum = torch.sum( beamformed, 1)

    beamformed_sum = beamformed_sum.permute([0 , 2 ,1])
    for i in range(beamformed_sum.size(0)):
        for j in range(128):
            beamformed_sum[i][j] = HilbertTransform(axis=0)(beamformed_sum[i][j])

    envelope = torch.abs(beamformed_sum)

      


    envelope = envelope.permute([0 ,2,1])
    image = 20*torch.log10(envelope/torch.max(envelope))
        
    bm_algo = p[0]

    loss = criterion(image, output_data)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()*image.size(0)       
    
    
    if math.isinf(train_loss) or math.isnan(train_loss):
        	print(f"Pass")
    else:
        print('Training Loss: {:.2f}'.format(train_loss))    
        strg = f"model/current_model"
        torch.save(model.state_dict() , strg)
        
        if bm_algo =="das":
            save_count(bm_algo)
        elif bm_algo == "dmas":
            save_count(bm_algo)
        elif bm_algo == "mvdr":
            save_count(bm_algo)
        else:
            save_count(bm_algo)


























