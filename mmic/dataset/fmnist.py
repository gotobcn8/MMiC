import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from .collector.tools import check, separate_data, split_data, save_file

def generate(dir_path,num_clients,num_classes,niid,balance,partition,alpha):
    if not os.path.isabs(dir_path):
        dir_path = os.path.join('repository',dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Setup directory for train/test data
    config_path = os.path.join(dir_path , "config.json")
    train_path = os.path.join(dir_path , "train/")
    test_path = os.path.join(dir_path , "test/")
    
    if check(config_path,train_path,test_path,num_clients,num_classes,niid,balance,partition,alpha):
       return
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)
    
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data
    
    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    
    X, y, statistic,overview = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition,alpha)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition,overview,alpha)