import torch
import torch.nn as nn 
import numpy as np 
import os



def render_images(model,test_datasets):
    #testing 
    model.eval()
    with torch.no_grad():
        ray