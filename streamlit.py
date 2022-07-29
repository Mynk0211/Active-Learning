#importing libraries
import streamlit as st
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
from function import *
from classes import *
import pickle
import time

st.title("Active Learning")



def plots(im1 , im2 , im3,im4):
  fig, ax = plt.subplots(1, 4, figsize=(12,2.5))
  fig.tight_layout()
  v_max =  0
  v_min = -70

  
  
  im = ax[0].imshow(im1 , aspect="auto" , cmap="gray" , vmin = v_min , vmax=v_max)
  ax[1].imshow(im2 , aspect="auto" , cmap="gray" , vmin = v_min , vmax=v_max)
  ax[2].imshow(im3 , aspect="auto" , cmap="gray" , vmin = v_min , vmax=v_max)
  ax[3].imshow(im4 , aspect="auto" , cmap="gray" , vmin = v_min , vmax=v_max)
  fig.colorbar(im, ax=ax.ravel().tolist())
  st.pyplot(fig)
  





    



def data_transfer(im1 , im2 , im3, im4 , input_data, model , optimizer , criterion):
    
    
    

   
	
    l = [("das" , im1) , ("dmas" , im2) , ("mvdr" , im3) , ("gcf" , im4)]  
	
    i = random.randint(0 , len(l) - 1)
    p1 = l[i]
    l.pop(i)
	
    i =random.randint(0 , len(l) - 1)
    p2 = l[i]
    l.pop(i)
    
    i =random.randint(0 , len(l) - 1)
    p3 = l[i]
    l.pop(i)
	
    p4 = l[0]
	
	
	
	
    plots(p1[1] , p2[1] , p3[1],p4[1])
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    

    with col1:
        if st.button(f"First"):
            s = time.time()
            training(model ,input_data , p1 , optimizer ,criterion )
            e= time.time()
            t = e-s
            print('Execution time: {:.4f}'.format(t)) 
            st.write("trained")
            
    with col2:
        if st.button(f"Second"):
            s = time.time()
            training(model ,input_data , p2 , optimizer ,criterion)
            e = time.time()
            t = e-s
            print('Execution time: {:.4f}'.format(t))
            st.write("trained")
    with col3:
        if st.button(f"Third "):
            s = time.time()
            training(model ,input_data , p3 , optimizer ,criterion)
            e = time.time()
            t = e-s
            print('Execution time: {:.4f}'.format(t))
            st.write("trained")
    with col4:
        if st.button(f"Fourth"):
            s = time.time()
            training(model ,input_data , p4 , optimizer ,criterion)
            e = time.time()
            t =e-s
            print('Execution time: {:.4f}'.format(t))
            st.write("trained")
            
            
        
    

    

def best_image(model ,optimizer , criterion,i):
    o1 = open("lists/DAS.pkl", "rb")
    l1 = pickle.load(o1)
    o1.close()
    
    o2 = open("lists/DMAS.pkl", "rb")
    l2 = pickle.load(o2)
    o2.close()
    
    o3 = open("lists/MVDR.pkl", "rb")
    l3 = pickle.load(o3)
    o3.close()
    
    o4 = open("lists/GCF.pkl", "rb")
    l4 = pickle.load(o4)
    o4.close()
    
    o5 = open("lists/data_cube.pkl", "rb")
    l5 = pickle.load(o5)
    o5.close()
    
    
    im1 = l1[i]
    im2 = l2[i]
    im3 = l3[i]
    im4 = l4[i]
    input_data = l5[i]


    data_transfer(im1, im2, im3, im4 , input_data, model , optimizer , criterion)
    
    

    



def main():
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = UNetLite()
    model.to(dev)
    optimizer = optim.Adam(model.parameters() , lr=0.001)
    criterion = nn.MSELoss()
    
    # i should range in between 0 to 4
    i  = 4
    
    
    if i ==0:
        p = "model/final_model"
    else:
        p = "model/current_model"
    
    try:
        model.load_state_dict(torch.load(p, map_location=dev)) 
    except:
        strg = f"model/final_model"
        torch.save(model.state_dict() , strg)
    

    best_image(model , optimizer , criterion,i)
    
    

    
main()
    



