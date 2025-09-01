import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import json

batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = 80 # guarantee that each client must have at least one samples for testing. 
# alpha = 0.1 # for Dirichlet distribution

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2,alpha=0.1):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    dataset_content, dataset_label = data

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
    print(len(X),len(X[0]))
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
    

# def separate_data_pre(data,num_clients,num_classes,niid=False,balance=False,partition=None,class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[i for i in range(num_classes)] for _ in range(num_clients)]
    
    dataset_content,dataset_label = data
    
    if not niid:
        parition = 'pat'
        class_per_client = num_classes
    if partition == 'pat':
        print('not realized')
        # idxs = np.array(range(len(dataset_label)))
        # idx_for_each_class = []
        # for i in range(len(num_classes)):
        #     idx_for_each_class.append(idxs[dataset_label == i])
        
        # class_num_per_client = [class_per_client for _ in range(num_clients)]
        # for i in range(num_clients):
        #     selected_clients = []
        #     for client in range(num_clients):
        #         if class_num_per_client > 0:
        #             selected_clients.append(client)
        #     selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)) * class_per_client)]
            
        #     num_all_samples = len(idx_for_each_class[i])
    if partition == 'dirichlet':
        alphas = [alpha] * num_clients
        #每个客户端获取1个类别的概率为alpha
        label_distribution = np.random.dirichlet(alphas,num_classes)
        y_class_index = [np.argwhere(dataset_label == y).flatten()
                     for y in range(num_classes)]
        #每个客户端对应的样本索引
        clients_dataset_map = [[] for _ in range(num_clients)]
        for class_k,fracs in zip(y_class_index,label_distribution):
            class_size = len(class_k)
            #如此得出类别k在每个一个客户端上能够划分的数量的数组
            splits = (fracs * class_size).astype(int)
            splits[-1] = class_size - splits[:1].sum()
            # class_k = np.random.shuffle(class_k)
            #np.split是根据边界来进行划分，比如三个客户端分别划分10,20,15个数据集，
            # 那么np.split输入的数组应该为[10,30,45]
            cumulative_sum = np.cumsum(splits)
            #idcs是个二维数组
            idcs = np.split(class_k,cumulative_sum)
            idcs = idcs[:-1]
            for i,idx in enumerate(idcs):
                clients_dataset_map[i].append(idx)
    
    for (i,client_i_data_idx) in enumerate(clients_dataset_map):
        for (j,class_k_index) in enumerate(client_i_data_idx):
            for index in class_k_index:
                X[i].append(dataset_content[index])
                y[i].append(dataset_label[index])
            # statistic is counting in client i how many dataset in class j?
            statistic[i][j] += len(class_k_index)   
            
            
    del data
    overview = dict()
    for client in range(num_clients):
        overview[client] = {
            # "labels":list(np.unique(y[client])),
            "labels": list(map(int,list(set(y[client])))),
            "data_size":len(X[client]),
        }
        print(f"Client {client}\t Size of data: {overview[client]['data_size']}\t Labels: ", overview[client]['data_size'])
        print(f"\t\t Samples of each labels: ", [i for i in statistic[client]])
        print("-" * 50)
    
    return X,y,statistic,overview
        

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None,overview=None,alpha=0.1):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        # 'Size of samples for labels in clients': statistic, 
        'overview':overview,
        'alpha': alpha, 
        'batch_size': batch_size, 
    }
    print(config)
    # gc.collect()
    print("Saving to disk.\n")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, str(idx) + '.npz'), 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path ,str(idx) + '.npz'), 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
    return config

def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        if len(X[i]) <= 4:
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True
        )

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        balance=True, partition=None,alpha = 0.1):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    # dir_path = os.path.dirname(train_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    # dir_path = os.path.dirname(test_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    return False

class OverView():
    def __init__(self,labels,size,client_id,statistic) -> None:
        self.cid = client_id
        self.labels = labels
        self.size = size
        self.statistic = statistic

if __name__ == '__main__':
    a = {
        'name':'fedAdvan',
        'overview':{
            'samples':[2,3,1,4],
            'size':78,
        },
        'time':458,
    }
    # tmp = {'num_clients': 20, 'num_classes': 10, 'non_iid': True, 'balance': False, 'partition': 'dirichlet', 'overview': {0: {'labels': [2, 3, 6, 7, 8], 'data_size': 240}, 1: {'labels': [1, 4, 6, 7, 8, 9], 'data_size': 1756}, 2: {'labels': [0, 1, 2, 3, 6, 7, 8, 9], 'data_size': 3486}, 3: {'labels': [0, 1, 2, 6, 7, 8, 9], 'data_size': 4779}, 4: {'labels': [1, 4, 5, 7, 8], 'data_size': 1129}, 5: {'labels': [5, 8], 'data_size': 856}, 6: {'labels': [0, 3, 7, 9], 'data_size': 5647}, 7: {'labels': [0, 1, 2, 3, 6, 9], 'data_size': 11313}, 8: {'labels': [0, 1, 2, 3, 4, 7, 9], 'data_size': 672}, 9: {'labels': [2, 4, 5, 9], 'data_size': 1449}, 10: {'labels': [1, 3, 4, 6, 7, 8, 9], 'data_size': 5697}, 11: {'labels': [1, 2, 3, 5, 7, 8], 'data_size': 1444}, 12: {'labels': [0, 2, 3, 6, 8], 'data_size': 5100}, 13: {'labels': [0, 1, 3, 4, 5, 9], 'data_size': 6660}, 14: {'labels': [1, 5, 6, 7, 8, 9], 'data_size': 3137}, 15: {'labels': [4, 5, 6, 8], 'data_size': 504}, 16: {'labels': [0, 2, 4, 5], 'data_size': 1129}, 17: {'labels': [0, 3, 6, 7], 'data_size': 353}, 18: {'labels': [0, 2, 3, 4, 6], 'data_size': 6496}, 19: {'labels': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'data_size': 8153}}, 'alpha': 0.1, 'batch_size': 10}
    values = list(np.array([1,7,3,6,8],dtype=int))
    with open('result.json', 'w') as f:
        json.dump(values, f)