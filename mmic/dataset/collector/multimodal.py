import os
import pickle
import random
import numpy as np

purpose = ['train','test','eval']

def get_flickr30k_data_type(data_path = './repository/flickr30k/',train_rate = 0.8,test_rate = 0.15,eval_rate = 0.05):
    '''
    We gonna convert captions.txt to form like this:
        train: [[img_path_1,caption_1],...,[img_path_n,caption_lentrain]]
        test: [[img_path_1,caption_1],...,[img_path_n,caption_lentest]]
        eval: [[img_path_1,caption_1],...,[img_path_n,caption_leneval]]
    '''
    if train_rate + test_rate + eval_rate > 1.0:
        AssertionError('train_rate + test_rate + eval_rate should be in 1.0')
    caption_data_file = os.path.join(data_path,'captions.txt')
    cap_file = open(caption_data_file)
    captions_list = []
    for line in cap_file.readlines():
        comma_split_values = line.split(',')
        del comma_split_values[1]
        captions_list.append(comma_split_values)
    random.shuffle(captions_list)
    dump_map = dict()
    total_length = len(captions_list)
    train_length,test_length,eval_length = int(total_length * train_rate),int(total_length * test_rate),int(total_length * eval_rate)
    dump_map['train'] = captions_list[:train_length]
    dump_map['test'] = captions_list[train_length:test_length+train_length]
    dump_map['eval'] = captions_list[test_length+train_length:test_length+train_length+eval_length]
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_path = os.path.join(data_path,'flickr30k_data.pkl')
    pickle.dump(dump_map,open(data_path, 'wb'))
    print('finished generate data type of flickr30k,path is {}'.format(data_path))
    

def seprate_noniid(root = './repository/flickr30k/',purpose_type = 0,num_clients = 15):
    '''
        purpose_type:
            0 -> train
            1 -> test
            2 -> eval
    '''
    pkl_path = os.path.join(root,'client_{}.pkl'.format(purpose[purpose_type]))
    data_type_path = os.path.join(root,'flickr30k_data.pkl')
    if not os.path.exists(data_type_path):
        get_flickr30k_data_type()
    data_purpose = pickle.load(open(data_type_path,'rb'))
    data = data_purpose[purpose[purpose_type]]
    
    if os.path.exists(pkl_path):
        dict_users = pickle.load(open(pkl_path, 'rb'))
    else:
        num_shards = 150
        num_imgs = int(len(data) / num_shards)
        idx_shard = [i for i in range(num_shards)]
        dict_users  ={i:np.array([],dtype=int) for i in range(num_clients)}
        idxs = np.arange(num_shards * num_imgs)
        img_idx = [i for i in range(len(data))]
        
        for i in range(num_clients):
            rand_set = set(np.random.choice(idx_shard, int(num_shards / num_clients), replace=False))  # idx_shardnum_shards/num_users
            idx_shard = list(set(idx_shard) - rand_set)  # Shards
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs: (rand+1)*num_imgs]), axis=0)
                img_idx = list(set(img_idx) - set(idxs[rand*num_imgs: (rand+1)*num_imgs]))
        
        dict_users[i] = np.concatenate([dict_users[i],img_idx])
        pickle.dump(dict_users, open(pkl_path, 'wb'))
    return dict_users