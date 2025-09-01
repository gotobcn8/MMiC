from .crisis_mmd import CrisisMMDGenerator as crisismmd_generate
from .flickr30k import generate as flickr30k_generate
from .food101 import Food101Generator as food101_generate
import os
alpha = 0.1

mm_dataset_generator = {
    'crisis_mmd': crisismmd_generate,
    'flickr30k': flickr30k_generate,
    'food101': food101_generate,
}

def download(name:str,niid,balance,partition,num_clients,num_classes,alpha):
    dir_path = name
    if partition == "-":
        partition = None
    
def multimodal_download(args):
    dataset_name = args.get('dataset')
    dataset_parameter = args.get(dataset_name)
    dataset_parameter['num_clients'] = args.get('num_clients')
    dataset_parameter['dataset'] = dataset_name
    dataset_parameter['device'] = args['device']
    mm_dataset_generator[dataset_name](dataset_parameter)
        
# def mm_dataset_download(dataset_name,tag):
#     mm_dataset_generator[dataset_name]()
    
# def multimodal_download(args,mm_dataset):
#     for key,dataset in args[mm_dataset].items():
#         if dataset in mm_dataset_generator.keys():
#             mm_dataset_download(dataset,key)
#         if dataset in traditional_dataset_generator.keys():
#             download(
#                 dataset,
#                 args[dataset]['niid'],
#                 args[dataset]['balance'],
#                 args[dataset]['partition'],
#                 args['client_settings'][key],
#                 args[dataset]['num_classes'],
#                 args[dataset]['alpha']
#             )