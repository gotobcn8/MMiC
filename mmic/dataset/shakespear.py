import numpy as np
import os
import sys
import random
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torchtext.vocab import set_default_index
from .collector.tools import separate_data,split_data,save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 4
max_len = 200
dir_path = 'shakespeare/'

def generate(dir_path,num_clients,num_classes,niid,balance,partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    path_collected = []
    config_path = os.path.join(dir_path,'config.json')
    train_path = os.path.join(dir_path,'train')
    test_path = os.path.join(dir_path,'test')
    raw_path = os.path.join(dir_path,'raw')
    dirs_collected = [train_path,test_path,raw_path]
    for dir in dirs_collected:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
