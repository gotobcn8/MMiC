from .agnews import generate as generate_agnews
from .mnist import generate as generate_mnist
from .fmnist import generate as generate_fmnist
from .cifar10 import generate_augmentation as generate_cifar10
from .emnist import generate as generate_emnist
from .coco import generate as generate_coco
from .flickr30k import generate as generate_flickr30k
from .multimodal.download import multimodal_download as new_multimodal_download
from .multimodal.download import mm_dataset_generator as new_mm_dataset_generator
import os
alpha = 0.1

mm_dataset_generator = {
    'coco' : generate_coco,
    'flickr30k' :generate_flickr30k,
}

traditional_dataset_generator = {
    'agnews':generate_agnews,
    'mnist':generate_mnist,
    'fmnist':generate_fmnist,
    'cifar10':generate_cifar10,
    'emnist':generate_emnist,
}

def download(name:str,niid,balance,partition,num_clients,num_classes,alpha):
    '''
    Normal dataset downloader.
    '''
    dir_path = name
    if partition == "-":
        partition = None
    if name in traditional_dataset_generator.keys():
        traditional_dataset_generator[name](dir_path,num_clients,num_classes, niid, balance, partition,alpha)
    
def multimodal_download(args,mm_dataset):
    '''
    Multimodal downloader
    '''
    if mm_dataset in new_mm_dataset_generator:
        new_multimodal_download(args)
        return
