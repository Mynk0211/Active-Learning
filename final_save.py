# -*- coding: utf-8 -*-

from classes import *
import torch




dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = UNetLite()
model.to(dev)

c_p = "model/current_model"
f_p = "model/final_model"

model.load_state_dict(torch.load(c_p, map_location=dev)) 

torch.save(model.state_dict() , f_p)