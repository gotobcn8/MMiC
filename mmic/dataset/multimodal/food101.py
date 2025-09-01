from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..collector.tools import check, separate_data, split_data, save_file
from sklearn.model_selection import train_test_split
import os
import ujson
import random
import pickle
import const.constants as const 
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils.downloader.pull import get_remote
import utils.path.unfile as unfile
from utils.path import create
least_samples = 300
CheckAttributeNames = ['num_clients','num_classes','non_iid','balance','partition','alpha']
FOOD101URL = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
class Food101Generator():
    def __init__(self,args) -> None:
        self.repodir = args['data_dir']
        self.config_path = os.path.join(self.repodir,'config.json')
        self.train_path = os.path.join(self.repodir,'train')
        self.test_path = os.path.join(self.repodir,'test')
        self.precheck()
        self.num_clients = args['num_clients']
        self.num_classes = args['num_classes']
        self.non_iid = args['niid']
        self.balance = args['balance']
        self.partition = args['partition']
        self.alpha = args['alpha']
        self.missing_modal_rate = args.get('missing_modal_rate')
        self.missing_modal_clients_rate = args.get('missing_modal_clients_rate')
        config,ok = self.check()
        if not ok:
            X,y,statistic,overview = self.data_partition(args)
            train_data,test_data = split_data(X,y)
            config = save_file(self.config_path, self.train_path, self.test_path, train_data, test_data, self.num_clients, self.num_classes, 
                statistic, self.non_iid, self.balance, self.partition,overview,self.alpha)
        # if config[]
        self.missing_check(config)
    
    # def precheck(self):
    #     if not os.path.exists(self.repodir):
    #         os.makedirs(self.repodir)
    #     if os.path.exists(self.config_path):
    #         print(f'{self.config_path} is exists, you can remove it if you want to re-generate a new data partition')
    #         return
    #     sz = os.path.getsize(self.repodir)
    #     if not sz:
    #         print('We start to download the directory')
    #         get_remote(
    #             url = FOOD101URL,
    #             save_path = os.path.join(self.repodir,'raw')
    #         )
    #         source_path = os.path.join(os.path.join(self.repodir,'raw'),FOOD101URL.split('/')[-1])
    #         print('Starting extract file...\n Plz waiting for a while')
    #         unfile.extract_tar_gz(source_path,os.path.join(self.repodir,'raw'))
    #     else:
    #         print(f'Your directory {self.repodir} is not NULL and without {self.config_path}, plz clean the directory firstly!')
    
    def precheck(self):
        create.makedirs(self.train_path)
        create.makedirs(self.test_path)
    
    def missing_check(self,config):
        overview = config['overview']
        if self.missing_modal_rate == config.get('missing_modal_rate') and self.missing_modal_clients_rate == config.get('missing_modal_clients_rate'):
            return
        missing_clients = random.sample(range(self.num_clients),int(self.missing_modal_clients_rate * self.num_clients))
        missing_clients_map = dict()
        for client in missing_clients:
            try:
                mssing_indexs = random.sample(range(overview[client]['data_size']),int(overview[client]['data_size'] * self.missing_modal_rate))
            except:
                mssing_indexs = random.sample(range(overview[str(client)]['data_size']),int(overview[str(client)]['data_size'] * self.missing_modal_rate))
            missing_idx_modal_set = dict()
            for index in mssing_indexs:
                #random select one modality to missing
                missing_idx_modal_set[index] = random.choice([0,1])
            missing_clients_map[client] = missing_idx_modal_set 

        if missing_clients_map is not None:
            if not os.path.exists(self.repodir):
                os.makedirs(self.repodir)
            with open(os.path.join(self.repodir, 'missing_' + str(self.missing_modal_rate) + '_' + str(self.missing_modal_clients_rate) + const.PICKLE_SUFFIX), 'wb') as f:
                # np.savez_compressed(f, data = missing_clients_map)
                pickle.dump(missing_clients_map,f)
        config['missing_modal_rate'] =  self.missing_modal_rate
        config['missing_modal_clients_rate'] = self.missing_modal_clients_rate
        with open(self.config_path, 'w') as f:
            ujson.dump(config, f)
        
    def check(self):
        if not os.path.exists(self.config_path):
            return None,False
        with open(self.config_path, 'r') as f:
            config = ujson.load(f)
        # if config['num_clients'] == self.num_clients and config.get('alpha',0.5) == self.alpha and config['partition'] == self.partition and config['balance'] == self.balance:
        #     return True,config
        for attr_name in CheckAttributeNames:
            if getattr(self,attr_name) != config.get(attr_name):
                return None,False
        return config,True

    def data_partition(self,args: dict):
        # Read arguments
        num_clients, alpha = args['num_clients'], args['alpha']
        num_classes = args.get('num_classes',101)
        prestore = args.get('prestore',False)
        # Define partition manager
        # pm = PartitionManager(args)
        
        # Fetch all labels
        # pm.fetch_label_dict() # obtaining the label dictionary 
        # get the raw csv data
        raw_data_path = Path(args['data_dir']).joinpath('raw','food101')
        texts_path = Path.joinpath(raw_data_path,'texts')
        train_images_path = Path.joinpath(raw_data_path,'images','train')
        test_images_path = Path.joinpath(raw_data_path,'images','test')
        train_texts_path = Path.joinpath(texts_path,'train_titles.csv')
        test_texts_path = Path.joinpath(texts_path,'test_titles.csv')
        
        # train dict generate from csv data
        train_data = pd.read_csv(train_texts_path,header = None)
        test_data = pd.read_csv(test_texts_path,header = None)
        class_names = dict()
        train_X,train_y = self.read_csv_data(data = train_data, images_path = train_images_path, class_names = class_names)
        test_X,test_y = self.read_csv_data(data = test_data, images_path = test_images_path, class_names = class_names)
        train_X,train_y = np.array(train_X),np.array(train_y)
        test_X,test_y = np.array(test_X),np.array(test_y)
        self.save_cluster_server_data(test_X=test_X,test_y=test_y)
        return classification_sepdata((train_X, train_y), num_clients, num_classes, 
                                        niid=args['niid'], balance = args['balance'], partition=args['partition'],alpha = args['alpha'])
        # return client_data_dict,num_classes

    def save_cluster_server_data(self,test_X,test_y):
        random_indices = np.random.choice(len(test_X), size=int(len(test_X) * 0.5), replace=False)
        test_X,test_y = test_X[random_indices],test_y[random_indices]
        X_cluster, X_server, y_cluster, y_server = train_test_split(
            test_X, test_y, train_size=0.5, shuffle=True
        )
        cluster_dict = {
            'x':X_cluster,
            'y':y_cluster,
        }
        server_dict ={
            'x':X_server,
            'y':y_server,
        }
        with open(os.path.join(self.test_path,  'cluster.npz'), 'wb') as f:
            np.savez_compressed(f, data=cluster_dict)
        with open(os.path.join(self.test_path,  'server.npz'), 'wb') as f:
            np.savez_compressed(f, data=server_dict)
        
    def read_csv_data(self,data,images_path,class_names):
        class_seq = 0
        train_X,train_y = [],[]
        for i in tqdm(range(len(data))):
            '''
            img_path = 'repository/food101/raw/apple_pie/xxx.img'
            '''
            caption,label = data.iloc[i][1],data.iloc[i][2]
            img_path = os.path.join(images_path,label,data.iloc[i][0])
            # img_path, caption, label = os.path.join(images_path,train_data.iloc[i][2],train_data.iloc[i][0]),train_data.iloc[i][1],train_data.iloc[i][2]
            train_X.append((img_path,caption))
            if label not in class_names:
                class_names[label] = class_seq
                class_seq += 1
            train_y.append(class_names[label])
        return train_X,train_y
        
def classification_sepdata(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2,alpha=0.1):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataset_content, dataset_label = data
    # dataset_label
    dataidx_map = {}
    
    if not niid:
        partition = 'pat'
        class_per_client = num_classes
    
    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dirichlet":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                # tmp = np.where(dataset_label == k)
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
    
    del data
    overview = dict()
    # print(len(X),len(X[0]))
    for client in range(num_clients):
        overview[client] = {
            # "labels":list(np.unique(y[client])),
            "labels": list(map(int,list(set(y[client])))),
            "data_size":X[client].shape[0],
        }
        print(f"Client {client}\t Size of data: {overview[client]['data_size']}\t Labels: ", overview[client]['data_size'])
        print(f"\t\t Samples of each labels: ", [i for i in statistic[client]])
        print("-" * 50)
    
    return X,y,statistic,overview
        

# def check(args)