import numpy as np
import torch.nn.functional as F
import torch

def get_normalized_value(power:int,times:int,floatPos:int = 4)->float:
    if times == 0:
        return 0
    return round(np.exp(2*power/times),floatPos)

def sigmoid(x,floatPos:int = 4)->float:
    return round(1 / (1+np.exp(-x)),floatPos)

def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)

def softplus(x,floatPos:int = 4) -> float:
    return round(np.log2(1 + np.exp(x)),floatPos)
# def tanh(x):

def OptRatio(x,floatPos:int = 4):
    '''
    This function is designed as normalized the optimal raito
    ### version 1: 
    2 + 2*np.exp(-x)) / (3 + np.exp(-x))
    ### version 2: 
    0.5 * (1 - np.exp(-0.2 * x))
    '''
    # return np.arctan(x+1)
    return np.tanh(x) / 10
    # return (1 / (1+np.exp(-(x+2))))

def maxmin_norm(x,maxv,minv,floatPos:int = 4):
    return (x - minv) / (maxv - minv + 1e-5)
