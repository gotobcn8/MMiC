import json
import pickle
import sys, os
import re
import argparse
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import const.constants as const
from container.mmloader import ImageTextSaver
from container.mmloader import ImageTextSaver
from .partition_manager import PartitionManager
import torch
import random
import ujson
# Define logging console
import logging
from ..collector.tools import check, separate_data, split_data, save_file

current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 container 目录的路径
container_dir = os.path.abspath(os.path.join(current_dir, 'container'))
# 添加 container 目录到 sys.path
sys.path.append(container_dir)

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    # re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return(text)

def data_partition(args: dict):
    
    # Read arguments
    num_clients, alpha = args['num_clients'], args['alpha']
    
    # Define partition manager
    pm = PartitionManager(args)
    
    # Fetch all labels
    pm.fetch_label_dict() # obtaining the label dictionary 
    # get the raw csv data
    data_path = Path(args['data_dir']).joinpath('raw','CrisisMMD_v2.0')
    train_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_train.tsv"), sep='\t')
    val_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_dev.tsv"), sep='\t')
    test_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_test.tsv"), sep='\t')

    train_data_dict, dev_data_dict, test_data_dict = dict(), dict(), dict()
    
    # train dict generate from csv data
    logging.info("Partition train data")
    for i in tqdm(np.arange(train_csv_data.shape[0])):
        train_text = remove_url(train_csv_data['tweet_text'].iloc[i]).strip()
        # print(train_text)
        train_data_dict[train_csv_data['image_id'].iloc[i]] = [
            train_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(train_csv_data['image'].iloc[i])),
            pm.label_dict[train_csv_data['label_image'].iloc[i]],
            train_text
        ]
        
    # val dict generate from csv data
    logging.info("Partition validation data")
    for i in tqdm(np.arange(val_csv_data.shape[0])):
        val_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
        dev_data_dict[val_csv_data['image_id'].iloc[i]] = [
            val_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(val_csv_data['image'].iloc[i])),
            pm.label_dict[val_csv_data['label_image'].iloc[i]],
            val_text
        ]
        
    # test dict generate from csv data
    logging.info("Partition test data")
    for i in tqdm(np.arange(test_csv_data.shape[0])):
        test_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
        test_data_dict[test_csv_data['image_id'].iloc[i]] = [
            test_csv_data['image_id'].iloc[i],
            str(Path(data_path).joinpath(test_csv_data['image'].iloc[i])),
            pm.label_dict[test_csv_data['label_image'].iloc[i]],
            test_text
        ]

    train_file_ids = list(train_data_dict.keys())
    train_file_ids.sort()
     
    file_label_list = [train_data_dict[file_id][2] for file_id in train_file_ids]
    
    # Perform split
    # file_idx_clients => [client0_file_idx: array, client1_file_idx: array, ...]
    file_idx_clients,K = pm.dirichlet_partition(
        file_label_list,
        min_sample_size=1
    )

    # Save the partition
    if 'output' not in args:
        args['output_dir'] = 'repository/crisis_mmd/'
    output_data_path = Path(args['output_dir']).joinpath('partition', args['dataset'])
    Path.mkdir(output_data_path, parents=True, exist_ok=True)

    # Obtrain train mapping
    client_data_dict = dict()
    for client_idx in range(num_clients):
        client_data_dict[client_idx] = [train_data_dict[train_file_ids[idx]] for idx in file_idx_clients[client_idx]]
    
    # Obtrain dev and test mapping
    client_data_dict["test_server"] = [dev_data_dict[file_id] for file_id in dev_data_dict]
    client_data_dict["test_cluster"] = [test_data_dict[file_id] for file_id in test_data_dict]
    num_classes = K
    return client_data_dict,num_classes


class CrisisMMDGenerator():
    def __init__(self,args) -> None:
        self.args = args
        self.repodir = args['data_dir']
        self.config_path = os.path.join(self.repodir,'config.json')
        self.train_path = os.path.join(self.repodir,'train')
        self.test_path = os.path.join(self.repodir,'test')
        # self.precheck()
        self.num_clients = args['num_clients']
        self.num_classes = args['num_classes']
        self.non_iid = args['niid']
        self.balance = args['balance']
        self.partition = args['partition']
        self.alpha = args['alpha']
        self.missing_modal_rate = args.get('missing_modal_rate')
        self.missing_modal_clients_rate = args.get('missing_modal_clients_rate')
        self.set_label_dict()
        config,overview = self.generate_dataset(args)
        if overview == None:
            print('generate failed, occurs errors in config file')
            return
        if not self.check_missing(
            config,
            self.missing_modal_rate,
            self.missing_modal_clients_rate,
        ):
            generate_missing_situations(args = args, config = config, overview = overview)
        
        with open(os.path.join('repository','crisis_mmd','config.json'), 'w') as f:
            ujson.dump(config, f)

    def data_partition(self,args: dict):
        # Read arguments
        num_clients, alpha = args['num_clients'], args['alpha']
        num_classes = self.num_classes
        prestore = args.get('prestore',False)
        # Define partition manager
        # pm = PartitionManager(args)
        
        # Fetch all labels
        # pm.fetch_label_dict() # obtaining the label dictionary 
        # get the raw csv data
        data_path = Path(args['data_dir']).joinpath('raw','CrisisMMD_v2.0')
        train_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_train.tsv"), sep='\t')
        val_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_dev.tsv"), sep='\t')
        test_csv_data = pd.read_csv(data_path.joinpath("crisismmd_datasplit_all", "task_humanitarian_text_img_test.tsv"), sep='\t')
        
        train_data_dict, dev_data_dict, test_data_dict = dict(), dict(), dict()
    
        # train dict generate from csv data
        logging.info("Partition train data")
        for i in tqdm(np.arange(train_csv_data.shape[0])):
            train_text = remove_url(train_csv_data['tweet_text'].iloc[i]).strip()
            # print(train_text)
            train_data_dict[train_csv_data['image_id'].iloc[i]] = [
                train_csv_data['image_id'].iloc[i],
                str(Path(data_path).joinpath(train_csv_data['image'].iloc[i])),
                self.label_dict[train_csv_data['label_image'].iloc[i]],
                train_text
            ]
            
        # val dict generate from csv data
        logging.info("Partition validation data")
        for i in tqdm(np.arange(val_csv_data.shape[0])):
            val_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
            dev_data_dict[val_csv_data['image_id'].iloc[i]] = [
                val_csv_data['image_id'].iloc[i],
                str(Path(data_path).joinpath(val_csv_data['image'].iloc[i])),
                self.label_dict[val_csv_data['label_image'].iloc[i]],
                val_text
            ]
            
        # test dict generate from csv data
        logging.info("Partition test data")
        for i in tqdm(np.arange(test_csv_data.shape[0])):
            test_text=remove_url(val_csv_data['tweet_text'].iloc[i]).strip()
            test_data_dict[test_csv_data['image_id'].iloc[i]] = [
                test_csv_data['image_id'].iloc[i],
                str(Path(data_path).joinpath(test_csv_data['image'].iloc[i])),
                self.label_dict[test_csv_data['label_image'].iloc[i]],
                test_text
            ]

        train_file_ids = list(train_data_dict.keys())
        train_file_ids.sort()
        
        file_label_list = [train_data_dict[file_id][2] for file_id in train_file_ids]
        
        # Perform split
        # file_idx_clients => [client0_file_idx: array, client1_file_idx: array, ...]
        file_idx_clients,K = self.dirichlet_partition(
            file_label_list,
            min_sample_size=100
        )
        
        # Save the partition
        if 'output' not in args:
            args['output_dir'] = 'repository/crisis_mmd/'
        output_data_path = Path(args['output_dir']).joinpath('partition', args['dataset'])
        Path.mkdir(output_data_path, parents=True, exist_ok=True)

        # Obtrain train mapping
        client_data_dict = dict()
        for client_idx in range(num_clients):
            client_data_dict[client_idx] = [train_data_dict[train_file_ids[idx]] for idx in file_idx_clients[client_idx]]
        
        # Obtrain dev and test mapping
        client_data_dict["test_server"] = [dev_data_dict[file_id] for file_id in dev_data_dict]
        client_data_dict["test_cluster"] = [test_data_dict[file_id] for file_id in test_data_dict]
        num_classes = K
        return client_data_dict,num_classes
    
    def set_label_dict(self)->int:
        self.label_dict = {
            'not_humanitarian':                         0, 
            'infrastructure_and_utility_damage':        1,
            'vehicle_damage':                           2, 
            'rescue_volunteering_or_donation_effort':   3,
            'other_relevant_information':               4, 
            'affected_individuals':                     5,
            'injured_or_dead_people':                   6, 
            'missing_or_found_people':                  7
        }

    def dirichlet_partition(
        self, 
        file_label_list: list,
        seed: int=8,
        min_sample_size: int=100
    ) -> (list):
        
        # cut the data using dirichlet
        min_size = 0
        K, N = len(np.unique(file_label_list)), len(file_label_list)
        # seed
        np.random.seed(seed)
        while min_size < min_sample_size:
            file_idx_clients = [[] for _ in range(self.args['num_clients'])]
            for k in range(K):
                idx_k = np.where(np.array(file_label_list) == k)[0]
                np.random.shuffle(idx_k)
                # if self.args.dataset == "hateful_memes" and k == 0:
                #    proportions = np.random.dirichlet(np.repeat(1.0, self.args.num_clients))
                # else:
                proportions = np.random.dirichlet(np.repeat(self.args['alpha'], self.args['num_clients']))
                # Balance
                proportions = np.array([p*(len(idx_j)<N/self.args['num_clients']) for p, idx_j in zip(proportions, file_idx_clients)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                file_idx_clients = [idx_j + idx.tolist() for idx_j,idx in zip(file_idx_clients,np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in file_idx_clients])
        return file_idx_clients,K
    
    def check(self,
        num_clients,
        alpha,
        partition,
        balance
    ):
        if not os.path.exists(self.config_path):
            return False,None
        with open(self.config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == self.num_clients and config.get('alpha',0.5) == self.alpha and config['partition'] == self.partition and config['balance'] == self.balance:
            return True,config
        # for attr_name in CheckAttributeNames:
        #     if getattr(self,attr_name) != config.get(attr_name):
        #         return None,False
        return False,config

    
    def generate_dataset(self,args):
        passed,config = self.check(
            num_clients=args['num_clients'],
            alpha = args['alpha'],
            partition=args['partition'],
            balance = args['balance'],
            # partition = 'Dirichlet'
        )
        if passed:
            return config,config['overview']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        client_data_dict,num_classes = self.data_partition(args)
        num_clients = args['num_clients']
        overview = dict()
        # with open('repository/crisis_mmd/config.json', 'wr') as f:
        #     config = ujson.load(f)
        for index,data_obj in client_data_dict.items():  
            save_type = const.PREFIX_TRAIN
            target = index
            if isinstance(index,str):
                # train_server
                indexs = index.split('_')
                save_type = indexs[0]
                target = indexs[-1]
            itloader = ImageTextSaver(
                index = target,
                dataset_obj = data_obj,
                device = device,
                save_path ='repository/crisis_mmd/'
            )
            itloader.save(save_type = save_type)
            overview[index] = {
                'size':len(itloader),
                'labels_static':itloader.y_static,
            }
        config = {
            'num_clients': num_clients, 
            'num_classes': num_classes, 
            'non_iid': True, 
            'balance': True, 
            'partition': 'dirichlet', 
            # 'Size of samples for labels in clients': statistic, 
            'overview':overview,
            'alpha': 0.5, 
        }
        # with open('repository/crisis_mmd/config.json', 'w') as f:
        #     ujson.dump(config, f)
        return config,overview

    def check_missing(
        self,
        config = None,
        mig_modal_rate = 0.0,
        mig_modal_cls_rate = 0.0
    ):
        if not config:
            return False
        if config.get('missing_modal_rate',0) == mig_modal_rate and config.get('missing_modal_clients_rate',0) == mig_modal_cls_rate:
            return True
        return False
    
# def check(
#     config_path = 'repository/crisis_mmd/config.json',
#     num_clients = 20,
#     alpha = 0.5,
#     # save_path,
#     partition = 'dirichlet',
#     is_balance = True,
# ):
#     if not os.path.exists(config_path):
#         return False,None
#     with open(config_path, 'r') as f:
#         config = ujson.load(f)
#     if config['num_clients'] == num_clients and config.get('alpha',0.5) == alpha and config['partition'] == partition and config['balance'] == is_balance:
#         return True,config
#     return False,None

# def generate(args):
#     config,overview = generate_dataset(args)
#     if overview == None:
#         # print('There is some errors about data generate overview')
#         # return
#         print('generate overview from config file')
#         overview = config.get('overview')
#         if overview == None:
#             print('generate failed, occurs errors in config file')
#             return
#     generate_missing_situations(args = args, config = config, overview = overview)
#     with open(os.path.join('repository','crisis_mmd','config.json'), 'w') as f:
#         ujson.dump(config, f)
    
# def generate_dataset(args):
#     passed,config = check(
#         num_clients=args['num_clients'],
#         alpha = args['alpha'],
#         partition=args['partition'],
#         is_balance = args['balance'],
#         # partition = 'Dirichlet'
#     )
#     if passed:
#         return config,None
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     client_data_dict,num_classes = data_partition(args)
#     num_clients = args['num_clients']
#     overview = dict()
#     # with open('repository/crisis_mmd/config.json', 'wr') as f:
#     #     config = ujson.load(f)
#     for index,data_obj in client_data_dict.items():  
#         save_type = const.PREFIX_TRAIN
#         target = index
#         if isinstance(index,str):
#             # train_server
#             indexs = index.split('_')
#             save_type = indexs[0]
#             target = indexs[-1]
#         itloader = ImageTextSaver(
#             index = target,
#             dataset_obj = data_obj,
#             device = device,
#             save_path ='repository/crisis_mmd/'
#         )
#         itloader.save(save_type = save_type)
#         overview[index] = {
#             'size':len(itloader),
#             'labels_static':itloader.y_static,
#         }
#     config = {
#         'num_clients': num_clients, 
#         'num_classes': num_classes, 
#         'non_iid': True, 
#         'balance': True, 
#         'partition': 'dirichlet', 
#         # 'Size of samples for labels in clients': statistic, 
#         'overview':overview,
#         'alpha': 0.5, 
#     }
#     # with open('repository/crisis_mmd/config.json', 'w') as f:
#     #     ujson.dump(config, f)
#     return config,overview

def generate_missing_situations(args:dict,config:dict,overview:dict):
    '''
    missing_clients_map:
    key: client id
    values: set([(dataset_index,modality_id),...])
    '''   
    mig_modality_rate,mig_modality_clients_rate = args.get('missing_modal_rate',0),args.get('missing_modal_clients_rate',0)
    missing_clients = random.sample(range(args['num_clients']),int(mig_modality_clients_rate * args['num_clients']))
    missing_clients_map = dict()
    for client in missing_clients:
        print(overview.keys)
        try:
            mssing_indexs = random.sample(range(overview[client]['size']),int(overview[client]['size'] * mig_modality_rate))
        except:
            mssing_indexs = random.sample(range(overview[str(client)]['size']),int(overview[str(client)]['size'] * mig_modality_rate))
        missing_idx_modal_set = dict()
        for index in mssing_indexs:
            #random select one modality to missing
            missing_idx_modal_set[index] = random.choice([0,1])
        missing_clients_map[client] = missing_idx_modal_set 

    if missing_clients_map is not None:
        if not os.path.exists(args['data_dir']):
            os.makedirs(args['data_dir'])
        with open(os.path.join(args['data_dir'], 'missing_' + str(mig_modality_rate) + '_' + str(mig_modality_clients_rate) + const.PICKLE_SUFFIX), 'wb') as f:
            # np.savez_compressed(f, data = missing_clients_map)
            pickle.dump(missing_clients_map,f)
    config['missing_modal_rate'] =  mig_modality_rate
    config['missing_modal_clients_rate'] = mig_modality_clients_rate

# def check_missing(
#     config_path = 'repository/crisis_mmd/config.json',
#     mig_modal_rate = 0.0,
#     mig_modal_cls_rate = 0.0
# ):
#     if not os.path.exists(config_path):
#         return False
#     with open(config_path, 'r') as f:
#         config = ujson.load(f)

#     if config.get('missing_modal_rate',0) == mig_modal_rate and config.get('missing_modal_clients_rate',0) == mig_modal_cls_rate:
#         return True
#     return False

if __name__ == "__main__":
    # read path config files
    path_conf = dict()
    # with open(str(Path(os.path.realpath(__file__)).parents[3].joinpath('system.cfg'))) as f:
    #     for line in f:
    #         key, val = line.strip().split('=')
    #         path_conf[key] = val.replace("\"", "")
            
    # If default setting
    path_conf['data_dir'] = 'repository/crisis_mmd/raw'
    # if path_conf["data_dir"] == ".":
    #     path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('data'))
    # if path_conf["output_dir"] == ".":
    #     path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[3].joinpath('output'))
    path_conf["output_dir"] = 'repository/crisis_mmd/'
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=path_conf["data_dir"],
        help="Raw data path of crisis-mmd data set",
    )
    
    parser.add_argument(
        "--output_partition_path",
        type=str,
        default=path_conf["output_dir"],
        help="Output path of crisis-mmd data set",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="federated",
        help="setup of the experiment: centralized/federated",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        '--num_clients', 
        type=int, 
        default=20, 
        help='Number of clients to cut from whole data.'
    )

    parser.add_argument(
        "--dataset",
        type=str, 
        default="crisis-mmd",
        help='Dataset name.'
    )
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client_data_dict,num_classes = data_partition(args)
    num_clients = args.num_clients
    clients_dataset_num = [0] * num_clients
    for index,data_obj in client_data_dict.items():
        itloader = ImageTextSaver(
            index = index,
            dataset_obj = data_obj,
            device = device,
            save_path ='repository/crisis_mmd/'
        )
        if index == 'val' or index == 'test':
            itloader.save(save_type = index)
        else:
            itloader.save(save_type='train')
        clients_dataset_num[index] = len(ImageTextSaver)
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': True, 
        'balance': True, 
        'partition': 'Dirichlet', 
        # 'Size of samples for labels in clients': statistic, 
        'overview':None,
        'alpha': 0.5, 
        # 'batch_size': 20, 
    }
    with open('repository/crisis_mmd/config.json', 'w') as f:
        ujson.dump(config, f)
    # ImageTextLoader