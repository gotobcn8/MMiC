from .client import ClientBase
from sklearn.preprocessing import label_binarize
import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics

class IFCAClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
    
    