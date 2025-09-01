import os
import json
# from loader import Crisis_MMD
from ..data import read_data
import torch
from torch.utils.data import DataLoader
from container.containerapi import Containers
# from 
GenralDataset = set(['crisis_mmd'])


def read_multimodal_data(dataset_name,client_sid,dataset_dir,data_type,is_preload = True):
    if is_preload:
        return load_multimodal_data(
            dataset_name = dataset_name,
            dataset_dir = dataset_dir,
            data_type=data_type,
            client_sid = client_sid,
        )
    else:
        '''
        Load raw data
        '''
        loaddata = load_raw_multimodaldata(
            dataset_name = dataset_name,
            dataset_dir = dataset_dir,
            data_type = data_type,
            client_sid = client_sid,
        )
        return Containers[dataset_name](loaddata[0],loaddata[1])
        

def load_multimodal_data(dataset_name,client_sid,dataset_dir,data_type):
    data_info = read_data(
            dataset = dataset_name,
            dir = dataset_dir,
            data_type = data_type,
            idx = client_sid,
        )
    X_data = solve_modality_data(data_info['x'])
    # X_data = torch.Tensor(data_info["x"]).type(torch.float32)
    y_data = torch.Tensor(data_info["y"]).type(torch.int64)
    dataloader = [(x, y) for x, y in zip(X_data, y_data)]
    return dataloader

def solve_modality_data(x_data):
    for i,x in enumerate(x_data):
        for j in range(len(x)-2):
            # if isinstance(x_data[i][j],int) or isinstance(x_data[i][j],float):
            #     x_data[i][j] = torch.Tensor([x_data[i][j]]).type(torch.float32)
            #     continue
            x_data[i][j] = torch.Tensor(x_data[i][j]).type(torch.float32)
    # x_data[i][2] = torch.Tensor(x_data[i][2]).type(torch.int)
    # x_data[i][3] = torch.Tensor(x_data[i][3]).type(torch.int)
    return x_data

def read_retrieval_data(dataset_name,client_sid,dataset_dir,data_type,is_preload):
    data_info = read_data(
        dataset = dataset_name,
        dir = dataset_dir,
        data_type = data_type,
        idx = client_sid,
    )
    X_data = solve_modality_data(data_info['x'])
    dataloader = [x for x in X_data]
    return dataloader

def load_raw_multimodaldata(dataset_name,client_sid,dataset_dir,data_type):
    data_info = read_data(
        dataset = dataset_name,
        dir = dataset_dir,
        data_type = data_type,
        idx = client_sid,
    )
    x_data = data_info['x']
    y_data = torch.as_tensor(data_info['y'],dtype = torch.int64)
    return (x_data,y_data)
