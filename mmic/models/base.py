from torch import nn
import torch
from typing import Union, List,Dict,Tuple

class ModelStone():
    def __init__(
        self, 
        models: Union[List,Dict,Tuple],
        optimizers: Union[List,Dict,Tuple],
        criterion: Union[List,Dict,Tuple],
        evaluator,
    ):
        self.models = models
        self.optimizers = optimizers
        self.criterion = criterion
        self.evaluator = evaluator
        
    def train(self,modelidx):
        self.models[modelidx].train()
    
    def forward(self,input,modelidx):
        pass

    def eval(self,modelidx):
        self.models[modelidx].eval()
    
    def test(self,input,modelidx):
        pass
        
    def size(self,modelidx):
        pass
      