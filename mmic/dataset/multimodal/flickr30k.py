import torch
import ujson
import os
import const.constants as const
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from container.retrieval import RetrievalSaver
import random

def data_partition(args,train_rate = 0.8,server_rate = 0.1,cluster_rate = 0.1,sub_rate = 0.05):
    '''
        purpose_type:
            0 -> train
            1 -> test
            2 -> eval
    '''
    # Input image ->output text
    # pkl_path = os.path.join(raw_data_dir,'client_{}.pkl'.format(purpose[purpose_type]))
    data_dir = args['data_dir']
    raw_data_dir = os.path.join(data_dir,'rawdata')
    images_path = os.path.join(raw_data_dir,'flickr30k-images')
    captions_path = os.path.join(raw_data_dir,'captions.txt')
    num_clients = args['num_clients']
    # generate captions list
    captions_list = []
    with open(captions_path,'r') as cap_file:
        for line in cap_file.readlines()[1:]:
            comma_split_values = line.split(',')
            del comma_split_values[1]
            captions_list.append(tuple(comma_split_values))
    # data length, we test used subrate to do it.
    data_length = len(captions_list)
    captions_list = captions_list[:int(data_length * sub_rate)]
    
    train_data,tmp_data = train_test_split(captions_list, test_size=(server_rate+cluster_rate), random_state=777)
    # second split in server and cluster
    server_test_data, cluster_test_data = train_test_split(tmp_data, test_size=server_rate, random_state=777)
    
    num_shards = 200
    num_each_shards = int(len(train_data) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    data_dict  ={i:np.array([],dtype=int) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_each_shards)
    train_idx = [i for i in range(len(train_data))]
    
    # clients_dict = dict() 
    train_captions_np = np.array(train_data)
    
    sum_total_num = 0
    pre_train_idx_length = len(train_idx)
    for i in range(num_clients):
        # example: from range(0,200) shards select 200/50 = 4 idxs
        rand_set = set(np.random.choice(idx_shard, int(num_shards / num_clients), replace=False))  # idx_shardnum_shards/num_users
        idx_shard = list(set(idx_shard) - rand_set)  # Shards
        current_client_nums = 0
        for rand in rand_set:
            rand_idxs = idxs[rand * num_each_shards:(rand+1) * num_each_shards]
            append_arrays = train_captions_np[rand_idxs]
            if len(data_dict[i]) == 0:
                data_dict[i] = append_arrays
            else:
                data_dict[i] = np.concatenate(
                    (data_dict[i],append_arrays),axis=0
                )
            # dict_users[i] = np.concatenate(
            #     (dict_users[i], idxs[rand*num_imgs: (rand+1)*num_imgs]), axis=0)
            train_idx = list(set(train_idx) - set(idxs[rand*num_each_shards: (rand+1)*num_each_shards]))
            # print('current train_idx length:{}'.format(pre_train_idx_length - len(train_idx)))
            current_client_nums += num_each_shards
        sum_total_num += current_client_nums
        # print(f'client {i} dataset num: {current_client_nums}, sum num: {sum_total_num}')
        
    data_dict[i] = np.concatenate([data_dict[i],train_captions_np[train_idx]],axis=0)
    data_dict['test_cluster'] = cluster_test_data
    data_dict['test_server'] = server_test_data
    return data_dict

# def generate(args):
#     generat_dataset

def generate_missing_situations(args:dict,config:dict,overview:dict):
    '''
    missing_clients_map:
    key: client id
    values: set([(dataset_index,modality_id),...])
    '''   
    mig_modality_rate,mig_modality_clients_rate = args.get('missing_modal_rate',0),args.get('missing_modal_clients_rate',0)
    if check_missing(mig_modal_rate = mig_modality_rate,mig_modal_cls_rate = mig_modality_clients_rate):
        return
    missing_clients = random.sample(range(args['num_clients']),int(mig_modality_clients_rate * args['num_clients']))
    missing_clients_map = dict()
    for client in missing_clients:
        try:
            mssing_indexs = random.sample(range(overview[client]['size']),int(overview[client]['size'] * mig_modality_rate))
        except:
            mssing_indexs = random.sample(range(overview[str(client)]['size']),int(overview[str(client)]['size'] * mig_modality_rate))
        missing_idx_modal_set = dict()
        for index in mssing_indexs:
            #random select one modality to missing
            missing_idx_modal_set[index] = random.choice([0,1])
        missing_clients_map[client] = missing_idx_modal_set 

    # missing_save_path = os.path.join(,'flickr30k')
    if missing_clients_map is not None:
        if not os.path.exists(args['data_dir']):
            os.makedirs(args['data_dir'])
        with open(os.path.join(args['data_dir'], 'missing_' + str(mig_modality_rate) + '_' + str(mig_modality_clients_rate) + const.PICKLE_SUFFIX), 'wb') as f:
            # np.savez_compressed(f, data = missing_clients_map)
            pickle.dump(missing_clients_map,f)
    config['missing_modal_rate'] =  mig_modality_rate
    config['missing_modal_clients_rate'] = mig_modality_clients_rate
    
def check_missing(
    config_path = 'repository/flickr30k/config.json',
    mig_modal_rate = 0.0,
    mig_modal_cls_rate = 0.0
):
    if not os.path.exists(config_path):
        return False
    with open(config_path, 'r') as f:
        config = ujson.load(f)

    if config.get('missing_modal_rate',0) == mig_modal_rate and config.get('missing_modal_clients_rate',0) == mig_modal_cls_rate:
        return True
    return False

def generate(args):
    config,overview = generate_dataset(args)
    if overview == None:
        # print('There is some errors about data generate overview')
        # return
        print('generate overview from config file')
        overview = config.get('overview')
        if overview == None:
            print('generate failed, occurs errors in config file')
            return
        
    generate_missing_situations(args = args, config = config, overview = overview)
    with open(os.path.join('repository','flickr30k','config.json'), 'w') as f:
        ujson.dump(config, f)
    
def generate_dataset(args):
    sub_rate = 0.5

    passed,config = check(
        num_clients=args['num_clients'],
        is_balance = args['balance'],
        niid=args['niid'],
        sub_rate = sub_rate,
        # partition = 'Dirichlet'
    )
    if passed:
        return config,None
    device = args['device'] if torch.cuda.is_available() else 'cpu'
    data_dict = data_partition(args = args,sub_rate = sub_rate)
    num_clients = args['num_clients']
    overview = dict()
    # with open('repository/crisis_mmd/config.json', 'wr') as f:
    #     config = ujson.load(f)
    for index,data_obj in data_dict.items():  
        save_type = const.PREFIX_TRAIN
        target = index
        if isinstance(index,str):
            # train_server
            indexs = index.split('_')
            save_type = indexs[0]
            target = indexs[-1]
        itloader = RetrievalSaver(
            index = target,
            dataset_obj = data_obj,
            device = device,
            save_path = args['data_dir'],
        )
        itloader.save(save_type = save_type)
        overview[index] = {
            'size':len(itloader),
            # 'labels_static':itloader.y_static,
        }
    config = {
        'num_clients': num_clients, 
        'non_iid': True, 
        'balance': True, 
        'partition': 'random_shards', 
        'overview':overview,
        'sub_rate':sub_rate,
        # 'batch_size': 20, 
    }
    return config,overview

def check(
    config_path = 'repository/flickr30k/config.json',
    num_clients = 20,
    # save_path,
    niid = True,
    is_balance = True,

    sub_rate = 0.05,
):
    if not os.path.exists(config_path):
        return False,None
    with open(config_path, 'r') as f:
        config = ujson.load(f)
    if config['num_clients'] == num_clients and config['non_iid'] == niid and config['balance'] == is_balance and sub_rate == config['sub_rate'] and config['overview'] is not None:
        return True,config
    return False,None