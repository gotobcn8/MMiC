from .client import ClientBase
from sklearn.preprocessing import label_binarize
import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
# from utils.data import copy as copys

class ClusterFL(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
        # generate a new 
        self.W = {key:value for key,value in self.model.named_parameters()}
        self.PreW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
    
    def copy(self,target, source):
        for name in target:
            target[name].data = source[name].data.clone()
    
    def subtract_(self,target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
    def reset(self): 
        self.copy(target=self.W, source=self.PreW)
    
    def cfl_train(self):
        self.copy(target=self.PreW,source=self.W)
        self.train()
        self.subtract_(target=self.dW, minuend=self.W, subtrahend=self.PreW)