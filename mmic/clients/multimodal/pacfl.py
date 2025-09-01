import utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import time
from .clientbase import MultmodalClientBase
from models.optimizer.ditto import PersonalizedGradientDescent
import copy
from algorithm.sim.lsh import ReflectSketch
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from operator import itemgetter
from collections import Counter
from utils.multimodaldata import api as loaderapi
import const.constants as const 
from container.processer import pretrain
import const.tasks as tasks 
import random
### Efficient Distribution Similarity Identification in Clustered Federated Learning via Principal Angles Between Client Data Subspaces

#### Introduction
'''
#### Distance Computing methods

Bhattacharyya Distance (BD), Maximum Mean Discrepancy
(MMD) (Gretton et al. 2012), and Kullback–Leibler (KL)
distance (Hershey and Olsen 2007).
'''
class PACFL(MultmodalClientBase):
    '''
    ### PACFL Introduction
    #### what is the limitations in prior Clustered Federated Learning?
    - Initial cluster models usually noisy.
    - It is time consuming until the  models training is stable.
    some jobs like **IFCA** (Ghosh et al. 2020) need to indicate the number of clustered groups. Leading to bad performance is possible.
    - Active models need to be downloaded in each round, this operation is costly in communication.
    - Couldn't balance well in Personalization and globalization.

    #### Target
    Thus, ***How a server can realize clustered FL efficiently by grouping the clients into clusters in a one-shot manner without requiring the number of clusters to be known automatically, but with substantially less communication cost?***

    #### Contributions
    - Computing a principle vector in different clients.
    - Our framework also naturally provides an elegant approach to handle newcomer clients unseen at training time by matching them with a cluster model that the client can further personalized with local training

    #### Preliminaries
    Using data distributions to measure similarity is tend to get better performance while model similarity usually will be influenced by bias and closed to each other.
    Using dataset in each client convert to left singular value to compute similarity
    '''
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
        self.algo = args['fedAlgorithm'][self.algorithm]
        self.budget = self.algo['budget']
        self.partition = 'dirichlet'
        self.nbias = 5

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=self.learning_rate_decay_gamma
        )   
        
    # def get_U_mask(self):
    #     class_samples,sorted_X = self.calculate_data_ratio()
    #     self.calculate_U_mask(class_samples,sorted_X)
    #     return self.U_mask
        
    # def calculate_data_ratio(self):
    #     # the sequence of train_data is messy
    #     train_data = loaderapi.DatasetGenerator[self.dataset](
    #         self.dataset,
    #         self.serial_id,
    #         self.dataset_dir,
    #         const.DataType[0],
    #         is_preload = self.is_data_preload
    #     )
    #     # train_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train = True)
    #     X_values,y_values = torch.stack([data[0][0] for data in train_data],dim=0),torch.stack([data[1] for data in train_data],dim=0)
    #     # sorted_indices = np.argsort(y_values)
    #     sorted_indices = torch.argsort(y_values)
    #     sorted_X = X_values[sorted_indices]
    #     sorted_y = y_values[sorted_indices]
    #     # unique_y,counts = np.unique(sorted_y,return_counts = True)
    #     unique_y,counts = torch.unique(sorted_y,return_counts = True)
    #     class_samples = {}
    #     for y,count in zip(unique_y,counts):
    #         class_samples[y] = count
    #     possess_class_nums = len(unique_y)
    #     base = 1 / possess_class_nums
    #     temp_ratio = {}
        
    #     for class_k in class_samples:
    #         proportion_k = class_samples[class_k]
    #         temp_ratio[class_k] = proportion_k
    #         if proportion_k >= (base+0.05):
    #             temp_ratio[class_k] = class_samples[class_k]
        
    #     sub_sum = sum(list(temp_ratio.values()))
        
    #     for class_k in temp_ratio.keys():
    #         temp_ratio[class_k] = (temp_ratio[class_k]/sub_sum)*self.budget
        
    #     round_ratio = self.round_to(list(temp_ratio.values()), self.budget)
    #     cnt = 0
    #     for class_k in temp_ratio.keys():
    #         temp_ratio[class_k] = round_ratio[cnt]
    #         cnt += 1
    #     self.train_count_ratio = temp_ratio 
    #     return class_samples,sorted_X
        
    # def calculate_U_mask(self,class_samples,sorted_X):
    #     cnt = 0
    #     U_temp = []
    #     K = 0
    #     for class_k,samples in class_samples.items():
    #         local_label_data = sorted_X[cnt:cnt+samples]
    #         local_label_data = torch.Tensor(local_label_data)
    #         local_label_data = local_label_data.reshape(samples.item(),-1).T
    #         # if type(sorted_y[cnt:cnt+samples]) == torch.Tensor:
    #         #     local_labels = list(set(sorted_y[cnt:cnt+samples].numpy()))
    #         # else:
    #         #     local_labels = list(set(sorted_y[cnt:cnt+samples]))
    #         if self.partition == 'dirichlet':
    #             if class_k in self.train_count_ratio.keys():
    #                 K = self.train_count_ratio[class_k]
    #             else:
    #                 K = self.nbias
            
    #         if K > 0:
    #             U1_temp,sh1_temp,vh1_temp = np.linalg.svd(local_label_data,full_matrices=False)
    #             U1_temp = U1_temp / np.linalg.norm(U1_temp,ord = 2,axis = 0)
    #             U_temp.append(U1_temp[:,0:K])
    #         cnt += samples

    #     self.U_mask = np.concatenate(U_temp,axis=1)  
    #     # print(self.U_mask.shape)  
    
    
    def get_U_mask(self):
        class_samples,sorted_X = self.calculate_data_ratio()
        self.calculate_U_mask(class_samples,sorted_X)
        return self.U_mask
        
    def calculate_data_ratio(self):
        # the sequence of train_data is messy
        train_data = loaderapi.DatasetGenerator[self.dataset](
            self.dataset,
            self.serial_id,
            self.dataset_dir,
            const.DataType[0],
            is_preload = self.is_data_preload
        )
        # train_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train = True)
        if not self.is_data_preload:
            X_values1,X_values2 = [data[0][0] for data in train_data],[data[0][1] for data in train_data]
            res = pretrain.after_clipprocessor((X_values1,X_values2))
            X_values = res['pixel_values']
        else:
            X_values = torch.stack([data[0][0] for data in train_data],dim=0)
        # HERE NEED TO BE FIX!! SHOULD REGULAR THE POSITION AS X,y,index
        # y_values = torch.stack([data[1] for data in train_data],dim=0)
        if self.task == tasks.TASK_RETRIEVAL:
            y_values = torch.Tensor([random.randint(0,5) for i in range(len(train_data))])
        else:
            y_values = torch.stack([data[-1] for data in train_data],dim=0)
        # sorted_indices = np.argsort(y_values)
        sorted_indices = torch.argsort(y_values)
        sorted_X = X_values[sorted_indices]
        sorted_y = y_values[sorted_indices]
        # unique_y,counts = np.unique(sorted_y,return_counts = True)
        unique_y,counts = torch.unique(sorted_y,return_counts = True)
        class_samples = {}
        for y,count in zip(unique_y,counts):
            class_samples[y] = count
        possess_class_nums = len(unique_y)
        base = 1 / possess_class_nums
        temp_ratio = {}
        
        for class_k in class_samples:
            proportion_k = class_samples[class_k]
            temp_ratio[class_k] = proportion_k
            if proportion_k >= (base+0.05):
                temp_ratio[class_k] = class_samples[class_k]
        
        sub_sum = sum(list(temp_ratio.values()))
        
        for class_k in temp_ratio.keys():
            temp_ratio[class_k] = (temp_ratio[class_k]/sub_sum)*self.budget
        
        round_ratio = self.round_to(list(temp_ratio.values()), self.budget)
        cnt = 0
        for class_k in temp_ratio.keys():
            temp_ratio[class_k] = round_ratio[cnt]
            cnt += 1
        self.train_count_ratio = temp_ratio 
        return class_samples,sorted_X
        
    def calculate_U_mask(self,class_samples,sorted_X):
        cnt = 0
        U_temp = []
        K = 0
        for class_k,samples in class_samples.items():
            local_label_data = sorted_X[cnt:cnt+samples]
            local_label_data = torch.Tensor(local_label_data)
            local_label_data = local_label_data.reshape(samples.item(),-1).T
            # if type(sorted_y[cnt:cnt+samples]) == torch.Tensor:
            #     local_labels = list(set(sorted_y[cnt:cnt+samples].numpy()))
            # else:
            #     local_labels = list(set(sorted_y[cnt:cnt+samples]))
            if self.partition == 'dirichlet':
                if class_k in self.train_count_ratio.keys():
                    K = self.train_count_ratio[class_k]
                else:
                    K = self.nbias
            
            if K > 0:
                U1_temp,sh1_temp,vh1_temp = np.linalg.svd(local_label_data,full_matrices=False)
                U1_temp = U1_temp / np.linalg.norm(U1_temp,ord = 2,axis = 0)
                U_temp.append(U1_temp[:,0:K])
            cnt += samples

        self.U_mask = np.concatenate(U_temp,axis=1)  
        # print(self.U_mask.shape)  
    
    def round_to(self,percents, budget=100):
        if not np.isclose(sum(percents), budget):
            raise ValueError
        n = len(percents)
        rounded = [int(x) for x in percents]
        up_count = budget - sum(rounded)
        errors = [(self.error_gen(percents[i], rounded[i] + 1) - self.error_gen(percents[i], rounded[i]), i) for i in range(n)]
        rank = sorted(errors)
        for i in range(up_count):
            rounded[rank[i][1]] += 1
        return rounded
    
    def error_gen(self,actual, rounded):
        divisor = np.sqrt(1.0 if actual < 1.0 else actual)
        return abs(rounded - actual) ** 2 / divisor