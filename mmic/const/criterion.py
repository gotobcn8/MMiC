import torch.nn as nn
from models.loss.ms import MCSoftContrastiveLoss
from models.loss.constrastive import ContrastiveLoss
from .tasks import TASK_RETRIEVAL

def get_celoss(task = None):
    return nn.CrossEntropyLoss()

def get_flickr30k(task = 'retrieval'):
    if task == 'retrieval':
        return ContrastiveLoss()

def GetDatasetLoss(dataset:str,task:int,device):
    if dataset == 'flickr30k':
        if task == TASK_RETRIEVAL:
            return ContrastiveLoss(device = device)
    else:
        return nn.CrossEntropyLoss()


