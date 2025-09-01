import copy
import numpy as np
import time
from .multimodalserver import Server
from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from models.optimizer.ditto import PersonalizedGradientDescent
from clients.multimodal.mmic import MMOFCHP as MMOFCHPClient
import time
from utils.data import read_client_data
# from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans,DBSCAN
from cluster.ofchpcluster import OFCHPCluster
import torch
import const.constants as const
from algorithm.augmentation import augmentation
import algorithm.normalized as norm
import random
from algorithm import randomnorm
from sklearn.metrics import silhouette_score
from utils.saver import save_as_type
from algorithm.sim.cluster import hierarchical_clustering
import os
from utils import vocab
import pickle
from models.multimodal.reconstruct import ImageToTextModel,TextToImageModel
import torch.optim as optim
import torch.nn as nn
from utils import saver
import json
from const.settings import Global

class MMiC(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.data_volume = self.fedAlgorithm['data_volume']
        self.hashF = SignRandomProjections(
            each_hash_num=self.fedAlgorithm['hash_num'],
            data_volume=self.data_volume,
            data_dimension=self.fedAlgorithm['cv_dim'],
            random_seed=args['random_seed']
        )
        self.augmentation = None
        if 'augmentation' in self.fedAlgorithm.keys():
            self.augmentation = self.fedAlgorithm['augmentation']
        self.fine_tuning_epoch = args['fine_tuning_epoch']
        self.sketches = dict()
        self.clients_ids_map = dict()
        self.set_ofchp_clients(MMOFCHPClient)
        # self.num_clusters = args['cluster']['cluster_num']
        # self.pre_pre_cluster(self.num_clusters)
        self.pre_cluster(args)
        self.set_clusters(OFCHPCluster,args)
        # This part is for pacfl
        
        self.set_for_new_clients()
        
        # switch on banzhaf power selection
        self.banzhaf_selection = args.get('selection',True)
        self.start_select_rounds = args.get('banzhaf_selected_rounds',5)
        
        self.max_score_gap = 0
        self.min_score_gap = 1 << 10
        self.select_prob = self.join_ratio
        self.__init_momentums_velocities()
        self.__init_visualized_collector()
        self.tau = 1e-3

        self.global_lr = args.get('global_learning_rate',1e-1)
        self.gen_modality_model(args)
        self.poverty_check = args.get('poverty_check',True)
        self.porfolio = args.get('porfolio',True)
        
    def gen_modality_model(self,args):
        self.is_gen_modality = args.get('is_gen_modality',False)
        if not self.is_gen_modality:
            return
        vocab_size = 49408
        i2t_model = ImageToTextModel(vocab_size=vocab_size).to(self.device)
        i2t_optimizer = optim.Adam(i2t_model.parameters(), lr=0.001)
        i2t_criterion = nn.CrossEntropyLoss(ignore_index=49407)
        args['i2t_model'] = i2t_model
        args['i2t_optimizer'] = i2t_optimizer
        args['i2t_criterion'] = i2t_criterion
        
        t2i_model = TextToImageModel(vocab_size = vocab_size).to(self.device)
        # Generate a random input text (batch_size=4, seq_len=10)
        t2i_optimizer = optim.Adam(t2i_model.parameters(), lr=0.001)
        t2i_criterion = F.mse_loss
        args['t2i_model'] = t2i_model
        args['t2i_optimizer'] = t2i_optimizer
        args['t2i_criterion'] = t2i_criterion
        # criterion = nn.mse
        
    # def _init_clients_select_func(self,select_rate,client_drop_rate):
    def set_clusters(self,clusterObj,args):
        self.clusters = []
        for i,cluster in enumerate(self.cluster_ids):
            self.clusters.append(clusterObj(args,i,cluster,self.cluster_map_clients[i]))

    def __init_momentums_velocities(self):
        self.momentums = {}
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            self.momentums[key] = torch.zeros_like(global_state_dict[key])
        # for key,param in self.global_model.named_parameters():
        #     self.momentums[key] = torch.zeros_like(param.data)
        self.velocities = copy.deepcopy(self.momentums)
    
    def set_ofchp_clients(self,clientObj):
        self.set_clients(clientObj)
        # return
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        for i,client in enumerate(self.all_clients):
            client.count_sketch(self.hashF)
            self.sketches[client.id] = client.minisketch
            self.clients_ids_map[client.id] = i
        self.slog.info('total calculating time {:.3f}s'.format(time.time() - start_time))

    def select_clients(self,i = 0):
        self.slog.info('Starting select clients for server')
        if self.banzhaf_selection and i >= self.start_select_rounds:
            self.get_cluster_probability()
            selected_clients = self.select_clients_by_power()
        else:
            selected_clients = self.norm_select_clients()
        # Use a cluster map the selected clients
        self.cluster_attend_clients = dict()
        for attend_client in selected_clients:
            cluster_id = self.clients_map_clusters[attend_client.id]
            if cluster_id not in self.cluster_attend_clients:
                self.cluster_attend_clients[cluster_id] = []
            #attender serial id in this round
            self.cluster_attend_clients[cluster_id].append(attend_client)
            self.each_clients_select_records[attend_client.serial_id] += 1
        return selected_clients
    
    def norm_select_clients(self):
        if self.random_clients_selected:
            # random number of attend clients
            self.current_num_join_clients = np.random.choice(int(self.num_original_clients * self.join_ratio),self.num_original_clients+1)
        else:
            # static number of attend clients
            self.current_num_join_clients = len(self.clients) * (1 - self.client_drop_rate) * self.join_ratio
        selected_clients = list(np.random.choice(self.clients,int(self.current_num_join_clients),replace=False))
        return selected_clients
    
    def select_clients_by_power(self):
        '''
        Updated 2nd Sep 2024, fix bug: imbalanced selected clients
        '''
        self.slog.info('Starting select clients for server')
        selected_clients = []
        oneclients_incluster = []
        for cluster in self.clusters:
            presum = 0
            if len(cluster.clients) <= 2:
                oneclients_incluster.extend(cluster.clients)
                continue
            for i in range(len(self.cluster_select_probabilities[cluster.id])-1):
                self.cluster_select_probabilities[cluster.id][i] = round(self.cluster_select_probabilities[cluster.id][i],4)
                presum += self.cluster_select_probabilities[cluster.id][i]
            self.cluster_select_probabilities[cluster.id][-1] = 1 - presum
            # for i,prob in enumerate(self.cluster_select_probabilities[cluster.id]):
            #     if randomnorm.select_by_probability(prob):
            #         selected_clients.append(cluster.clients[i])
            cluster_selected_size = int(len(cluster.clients) * self.join_ratio)
            selected_clients.extend(np.random.choice(a = cluster.clients,size = cluster_selected_size,p=self.cluster_select_probabilities[cluster.id],replace=False))
        # except_drop_nums = int(len(selected_clients) * (1 - self.client_drop_rate))
        # selected_clients = list(set(selected_clients))
        #select only one client cluster
        if len(oneclients_incluster):
            cluster_selected_size = int(len(oneclients_incluster) * self.join_ratio)
            if cluster_selected_size == 0:
                cluster_selected_size = 1
            selected_clients.extend(np.random.choice(a = oneclients_incluster,size = cluster_selected_size,replace=False))
        if len(selected_clients) > self.current_num_join_clients:
            selected_clients = np.random.choice(a = selected_clients, size = int(self.current_num_join_clients),replace=False)
        selected_clients = list(selected_clients)
        while len(selected_clients) < self.current_num_join_clients:
            client = random.choice(self.clients)
            if client not in selected_clients:
                selected_clients.append(client)
        # selected_clients = list(np.random.choice(selected_clients,except_drop_nums,replace=False))
        return selected_clients
    
    def __init_visualized_collector(self):
        self.important_layers_by_rounds = []
        self.missing_modalities_by_rounds = []
    
    def train(self):
        new_attend_clients = []
        self.slog.debug('server','starting to train')
        is_new_joined = False
        for i in Global.iterator():
            s_t = time.time()
            self.selected_clients = self.select_clients(i)
            self.send_models()
            self.slog.info('total {} clients selected in this round'.format(len(self.selected_clients)))
            # if i % self.eval_gap == 0:
            #     self.cluster_gen_train()
            for client in self.selected_clients:
                client.global_rounds = i
                client.train()
            
            if (i-1) % self.eval_gap == 0:
                self.slog.debug('clients selected times:{}'.format(self.each_clients_select_records))
                self.slog.debug(f"-------------Round number for evaluate: {i}-------------")
                self.ofchp_validate()
                self.attend_clients_validate()
            # whether there are late clients still existed.
            if is_new_joined:
                # If there are some new clients join, need to test the generalization firstly.
                self.evaluate_generalized(new_attend_clients)          
            # if i % self.eval_gap == 0:
            #     self.cluster_validate()
            # self.receive_models()
            self.cluster_receive_models()
            if self.poverty_check:
                self.cluster_avg_important_parameter()
            self.cluster_aggregate_parameters()
            # self.aggregate_parameters()
            self.update_parameters()
            
            if i % self.eval_gap == 0:
                self.cluster_gen_train()
                self.cluster_validate()
            # self.check_global_model()
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            self.tracker.clear_cache()
            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
            if self.new_clients_settings['enabled'] and i+1 >= self.start_new_joining_round and len(self.late_clients) > 0:
                if len(self.late_clients) < self.num_new_join_each_round:
                    new_attend_clients = self.late_clients
                    self.late_clients = []
                else:
                    new_attend_clients = self.late_clients[:self.num_new_join_each_round]
                    self.late_clients = self.late_clients[self.num_new_join_each_round:]
                #it need to be fine-tuned before attending
                self.fine_tuning_new_clients(new_attend_clients)
                self.clients.extend(new_attend_clients)
                is_new_joined = True
            else:
                is_new_joined = False
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc)) 
        self.save_collector()
        self.cluster_save()
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))

        # self.save_running_process()
        self.save_results()
        self.save_global_model()

    def cluster_save(self):
        important_count = []
        client_selected_times = []
        for i in range(self.num_clusters):
            important_count.append(self.clusters[i].importance_count)
            client_selected_times.append(self.clusters[i].clients_select_times)
        cluster_collector ={
            'important_count': important_count,
            'client_selected_times': client_selected_times,
        }
        saver.save_as_type(cluster_collector,file_name='cluster_collector',data_path=self.save_dir,save_type='pickle')
        self.slog.info('cluster info save in {}'.format(os.path.join(self.save_dir,'cluster_collector.pkl')))
        
    def cluster_gen_train(self):
        if not self.is_gen_modality:
            return
        train_loader = self.load_dataset(data_type = const.PREFIX_TEST,shuffle = False, batch_size=10,index = 'cluster')
        self.args['i2t_model'].train()
        self.args['t2i_model'].train()
        for _ in range(5):
            for i,data in enumerate(train_loader):
                (img,text,_),label,_ = data
                img = img.to(self.device)
                text = text.to(self.device)
                # print(text.shape)
                lengths = vocab.get_lengths(text)
                # lengths_device = lengths.to(self.device)
                # print(lengths)
                i2t_outputs = self.args['i2t_model'](img, text[:, :-1],lengths)  # Remove last word for input
                i2t_loss = self.args['i2t_criterion'](i2t_outputs.reshape(-1, i2t_outputs.shape[2]), text[:, 1:].reshape(-1))
                self.args['i2t_optimizer'].zero_grad()
                i2t_loss.backward()
                self.args['i2t_optimizer'].step()

                lengths_device = torch.as_tensor(lengths)
                lengths_device = lengths_device.to(self.device)
                outputs = self.args['t2i_model'](text,lengths_device)  # Remove last word for input
                loss = self.args['t2i_criterion'](outputs,img)
                self.args['t2i_optimizer'].zero_grad()
                loss.backward()
                self.args['t2i_optimizer'].step()

    # def aggregate_parameters(self):
    #     global_state_dict = copy.deepcopy(self.global_model.state_dict())
    #     for param in self.global_model.parameters():
    #         param.data.zero_()
    #     for cluster_id,weight in self.clusters_weights.items():
    #         self.add_parameters(global_state_dict,weight,cluster_id)

    # def add_parameters(self,global_state_dict,weight,cluster_id):
    #     cluster_model = self.clusters[cluster_id].cluster_model
    #     cluster_state_dict = cluster_model.state_dict()
    #     ratio = 1.0
    #     if self.cluster_optimal_portfolio.get(cluster_id,0) > 0:
    #         x = self.cluster_optimal_portfolio[cluster_id]
    #         ratio = norm.OptRatio(x)
    #     cur_global_state_dict = self.global_model.state_dict()
    #     for key in global_state_dict:
    #         cur_global_state_dict[key] += global_state_dict[key] + (cluster_state_dict[key] - global_state_dict[key]) * weight * ratio
    #     self.global_model.load_state_dict(cur_global_state_dict)
        # for server_param,cluster_param in zip(self.global_model.parameters(),cluster_model.parameters()):
        #     # server_param.data += client_param.data.clone() * weight
        #     #ratio = 1 / 1+log(opt)
        #     ratio = 1.0
        #     if self.cluster_optimal_portfolio.get(cluster_id,0) > 0:
        #         x = self.cluster_optimal_portfolio[cluster_id]
        #         # ratio = (1 / (1+min(np.log10(),0)))
        #         # ratio = (1 / (1 + np.tanh(x)))
        #         ratio = norm.OptRatio(x)
        #     server_param.data += (cluster_param.data - server_param.data) * weight * ratio
    @torch.no_grad()
    def aggregate_parameters(self):
        global_state_dict = self.global_model.state_dict()
        pre_global_state_dict = copy.deepcopy(global_state_dict)
        # for param in self.global_model.parameters():
        #     param.data.zero_()
        for key in global_state_dict:
            global_state_dict[key].data.zero_()
        for cluster_id,weight in self.clusters_weights.items():
            # self.add_parameters(global_state_dict,weight,cluster_id)
            cluster_state_dict = self.clusters[cluster_id].cluster_model.state_dict()
            ratio = 1.0
            if self.cluster_optimal_portfolio.get(cluster_id,0) > 0:
                x = self.cluster_optimal_portfolio[cluster_id]
                ratio = norm.OptRatio(x)
            for key in global_state_dict:
                global_state_dict[key] += (pre_global_state_dict[key] * (1 - ratio) + (cluster_state_dict[key]) * ratio) * weight
        self.global_model.load_state_dict(global_state_dict)
    
    # @torch.no_grad()
    # def update_parameters(self):
    #     # self.old_global_model = copy.deepcopy(self.global_model)
    #     weighted_params_diff = {}
    #     global_state_dict = self.global_model.state_dict()
    #     Mbeta = 0
    #     for i in self.cluster_attend_clients:
    #         cluster_state_dict = self.clusters[i].cluster_model.state_dict()
    #         for key in global_state_dict:
    #             weighted_params_diff[key] = weighted_params_diff.get(key,0) + (self.clusters_weights[i] * (global_state_dict[key] - cluster_state_dict[key]))
    #         Mratio = 0
    #         if self.cluster_optimal_portfolio.get(i,0) > 0:
    #             Mratio = self.cluster_optimal_portfolio[i]
    #             # Mratio = norm.OptRatio(x)
    #         Mbeta += Mratio * self.clusters_weights[i]

    #     if not self.porfolio:
    #         Mbeta = 0
    #     Mbeta = 0.9 + norm.OptRatio(Mbeta)
    #     print('current Mbeta:',Mbeta)
    #     for key in self.momentums:
    #         self.momentums[key] = Mbeta * self.momentums[key] + (1 - Mbeta) * weighted_params_diff[key]
        
    #     self._update_adagrad(weighted_params_diff)
        
    #     for key in self.momentums:
    #         global_state_dict[key] = global_state_dict[key] - self.global_lr * (self.momentums[key] / (self.velocities[key].sqrt() + self.tau))

        
    #     self.global_model.load_state_dict(global_state_dict,strict = True)

    @torch.no_grad()
    def update_parameters(self):
        # self.old_global_model = copy.deepcopy(self.global_model)
        weighted_params_diff = {}
        global_state_dict = self.global_model.state_dict()

        for i in self.cluster_attend_clients:
            cluster_state_dict = self.clusters[i].cluster_model.state_dict()
            mbeta = self.cluster_optimal_portfolio.get(i,0)
            if not self.porfolio:
                Mbeta = 0
            Mbeta = 0.9 + norm.OptRatio(mbeta)
            for key in self.momentums:
                self.momentums[key] += (1 - Mbeta) * self.clusters_weights[i] * (global_state_dict[key] - cluster_state_dict[key])
                weighted_params_diff[key] = weighted_params_diff.get(key,0) + (self.clusters_weights[i] * (global_state_dict[key] - cluster_state_dict[key]))
                
        self._update_adagrad(weighted_params_diff)
        
        for key in self.momentums:
            global_state_dict[key] = global_state_dict[key] - self.global_lr * (self.momentums[key] / (self.velocities[key].sqrt() + self.tau))
        
        self.global_model.load_state_dict(global_state_dict,strict = True)

    
    def _update_adagrad(self, delta_list):
        for key in self.velocities:
            self.velocities[key] = self.velocities[key] + delta_list[key] ** 2

    def add_parameters(self,global_state_dict,weight,cluster_id):
        cluster_model = self.clusters[cluster_id].cluster_model
        cluster_state_dict = cluster_model.state_dict()
        ratio = 1.0
        if self.cluster_optimal_portfolio.get(cluster_id,0) > 0:
            x = self.cluster_optimal_portfolio[cluster_id]
            ratio = norm.OptRatio(x)
        new_global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            new_global_state_dict[key] += global_state_dict[key] * (1 - ratio) + (cluster_state_dict[key]) * weight * ratio
        self.global_model.load_state_dict(new_global_state_dict)
    
    def send_models(self):
        assert (len(self.clients) > 0)
        for cluster in self.clusters:
            cluster.set_parameters(self.global_model)
            for client in cluster.clients:
                start_time = time.time()
                client.set_parameters(cluster.cluster_model)
                client.send_time['rounds'] += 1
                client.send_time['total_cost'] += 2 * (time.time() - start_time)
            
    def cluster_receive_models(self):
        self.clusters_weights = dict()
        total_cluster_samples = 0 
        for cluster_id,attend_clients in self.cluster_attend_clients.items():
            self.clusters[cluster_id].receive_models(attend_clients)
            for attend_client in attend_clients:
                self.clusters_weights[cluster_id] = self.clusters_weights.get(cluster_id,0) + attend_client.train_samples
                total_cluster_samples += attend_client.train_samples
        
        for cluster_id in self.clusters_weights:
            self.clusters_weights[cluster_id] /= total_cluster_samples          
        # assert 1 - sum(self.clusters_weights) < 1e-3

    def pre_cluster_dbscan(self, eps=1.85, min_samples=5):
        sketches1dim = []
        cluster_start_time = time.time()
        # sketch_saver = []
        
        # Collect all client sketches
        for client in self.all_clients:
            sketch = client.minisketch
            sketches1dim.append(sketch.reshape(1, -1)[0])
            # sketch_saver.append(sketch)
        
        # Save the sketches
        saver.save_as_type(
            data=sketches1dim,
            file_name='sketches',
            data_path='repository/',
            mode='wb'
        )
        
        # Convert sketches to numpy array for DBSCAN
        sketches1dim = np.array(sketches1dim)
        
        # Apply DBSCAN, eps and min_samples control the clustering behavior
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_res = dbscan.fit(sketches1dim)
        
        self.slog.info('server,cluster time:{:.3f}s'.format(time.time() - cluster_start_time))
        
        # Initialize mappings
        self.clients_map_clusters = dict()
        unique_labels = np.unique(dbscan_res.labels_)
        
        # Exclude noise (-1 label)
        self.num_clusters = len(unique_labels[unique_labels != -1])
        
        self.cluster_map_clients = [[] for _ in range(self.num_clusters)]
        self.cluster_ids = [[] for _ in range(self.num_clusters)]
        
        for client_sid, cluster_id in enumerate(dbscan_res.labels_):
            if cluster_id == -1:
                # Ignore noise points
                continue
            self.clients_map_clusters[self.all_clients[client_sid].id] = cluster_id
            self.cluster_map_clients[cluster_id].append(self.all_clients[client_sid])
            self.cluster_ids[cluster_id].append(client_sid)
        
        # Print cluster information
        print('-' * 25)
        for i, c in enumerate(self.cluster_ids):
            print(f'cluster {i}: {c}')
            print('-' * 25)

    
    def pre_cluster(self,args):
        cluster_param = args['cluster']
        cluster_file = cluster_param.get('cluster_file')
        if cluster_file is not None:
            with open(cluster_file,mode='r') as cf:
                cluster_res = json.load(cf)
            # key = 'pacfl_cluster_17'
            key = 'manually'
            self.cluster_by_maninput(cluster_res[key])
            return
        cluster_algo_name = cluster_param.get('algo','ofchp')
        if cluster_algo_name == 'ofchp':
            self.num_clusters = cluster_param.get('cluster_num',10)
            self.ofchp_pre_cluster(self.num_clusters)
        elif cluster_algo_name == 'pacfl':
            self.cluster_alpha = self.fedAlgorithm['alpha']
            self.linkage = self.fedAlgorithm['linkage']
            self.cluster_by_Umasks()
    
    def cluster_by_maninput(self,format):
        self.clients_map_clusters = dict()
        if type(format) == list:
            self.num_clusters = len(set(format))
            self.cluster_map_clients = [[] for _ in range(self.num_clusters)]
            self.cluster_ids = [[] for _ in range(self.num_clusters)]
            for client_sid,cluster_id in enumerate(format):
                self.clients_map_clusters[self.all_clients[client_sid].id] = cluster_id
                self.cluster_map_clients[cluster_id].append(self.all_clients[client_sid])
                self.cluster_ids[cluster_id].append(client_sid)
        elif type(format) == dict:
            self.num_clusters = len(format)
            self.cluster_map_clients = [[] for _ in range(self.num_clusters)]
            self.cluster_ids = [[] for _ in range(self.num_clusters)]
            for cluster_id,clients in format.items():
                cluster_id = int(cluster_id)
                for client_sid in clients:
                    self.clients_map_clusters[self.all_clients[client_sid].id] = cluster_id
                    self.cluster_ids[cluster_id].append(client_sid)
                    self.cluster_map_clients[cluster_id].append(self.all_clients[client_sid])

    def ofchp_pre_cluster(self,cluster_num):
        sketches1dim = []
        cluster_start_time = time.time()
        sketch_saver = []
        for client in self.all_clients:
            sketch = client.minisketch
            # print(np.count_nonzero(sketch))
            sketches1dim.append(sketch.reshape(1,-1)[0]) 
            sketch_saver.append(sketch)
        kmeans = KMeans(n_clusters=cluster_num,random_state=666)
        saver.save_as_type(
            data = sketch_saver,
            file_name='sketches',
            data_path='repository/',
            mode='wb'
        )
        #fit the kmeans with the one-dimensional data to cluster
        kmeans_res = kmeans.fit(sketches1dim)
        self.slog.info('server,cluster time:{:.3f}s'.format(time.time() - cluster_start_time))
        '''
        from the cluster form a map to the {client_id:cluster_id}
        from the cluster form a map to the {cluster id: [clients_obj]}
        '''
        final_clusters = []
        self.clients_map_clusters = dict()
        self.cluster_map_clients = [[] for _ in range(self.num_clusters)]
        self.cluster_ids = [[] for _ in range(self.num_clusters)]
        for client_sid,cluster_id in enumerate(kmeans_res.labels_):
            self.clients_map_clusters[self.all_clients[client_sid].id] = cluster_id
            self.cluster_map_clients[cluster_id].append(self.all_clients[client_sid])
            self.cluster_ids[cluster_id].append(client_sid)
        
        print('-'*25)
        for i,c in enumerate(self.cluster_ids):
            print('cluster {}:{}'.format(i,c))
            print('-'*25)
        
        # silhouette_score(sketches1dim,kmeans_res.labels_,metric='euclidean')
        # data_path = 'cluster'
        # save_as_type(sketches1dim,file_name='clusterdata',data_path=data_path)
        
    def fine_tuning_new_clients(self,new_clients):
        for new_client in new_clients:
            which_cluster = self.clients_map_clusters[new_client.id]
            new_client.set_parameters(self.clusters[int(which_cluster)].cluster_model)
            # new_client.model = copy.deepcopy(self.clusters[int(which_cluster)].cluster_model)
            optimizer = torch.optim.SGD(new_client.model.parameters(),lr = self.learning_rate)
            lossFunc = torch.nn.CrossEntropyLoss()
            train_loader = new_client.load_train_data()
            new_client.model.train()
            for _ in range(self.fine_tuning_epoch):
                for _,(x,y) in enumerate(train_loader):
                    if isinstance(x,list):
                        x[0] = x[0].to(new_client.device)
                        x = x[0]
                    else:
                        x = x.to(new_client.device)
                    y = y.to(new_client.device)
                    output = new_client.model(x)
                    loss =  lossFunc(output,y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            new_client.model_person = copy.deepcopy(new_client.model)
            new_client.optimizer_personl = PersonalizedGradientDescent(
                new_client.model_person.parameters(), lr=new_client.learning_rate, mu=new_client.mu)
    
    def test_metrics_personalized(self):
        # return super().test_metrics()
        num_samples = []
        total_corrects = []
        total_auc = []
        ids = [0] * len(self.clients)
        for i,c in enumerate(self.clients):
            # if c.id.startswith('late'):
            #     self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
            c_corrects,c_num_samples,c_auc = c.test_metrics()
            total_corrects.append(c_corrects*1.0)
            total_auc.append(c_auc * c_num_samples)
            num_samples.append(c_num_samples)
            ids[i] = c.id
        
        return ids,num_samples,total_corrects,total_auc
    
    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0],[1],[0]
        num_samples = []
        losses = []
        for c in self.clients:
            cl,ns = c.train_personalized_with_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        
        return ids,num_samples,losses
    
    def evaluate_generalized(self,new_attend_clients=[]):
        for _,c in enumerate(new_attend_clients):
            self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
    
    def cluster_aggregate_parameters(self):
        # for cluster in self.clusters:
        #     cluster.aggregate_parameters()
        if len(self.clusters) <= 0:
            self.slog.info('No clusters in server')
            return
        
        for cluster_id in self.cluster_attend_clients.keys():
            self.clusters[cluster_id].aggregate_parameters()
    
    @torch.no_grad()
    def cluster_avg_important_parameter(self):
        # client_corresponding_parameters = {}
        clients_missing_map = {}
        clients_important_layers_map = {}
        for client in self.selected_clients:
            if client.important_layer_key == '':
                continue
            avg_param = torch.zeros_like(client.model.state_dict()[client.important_layer_key])
            cur_samples = 0
            # To find the corresponding cluster
            cluster = self.clusters[self.clients_map_clusters[client.id]]
            # do the layer fedavg
            self.slog.debug('client {} effected by layer {} and missing modalities:{}'.format(client.serial_id,client.important_layer_key,client.train_missing_modalities))
            clients_missing_map[client.serial_id] = client.train_missing_modalities
            clients_important_layers_map[client.serial_id] = client.important_layer_key
            if len(cluster.clients) < 2:
                continue
            for c in cluster.clients:
                if client.important_layer_key != c.important_layer_key:
                    cur_samples += c.train_samples
                    avg_param += (c.train_samples * c.model.state_dict()[client.important_layer_key]) 
            avg_param /= cur_samples
            client.model.state_dict()[client.important_layer_key].copy_(avg_param)
        self.important_layers_by_rounds.append(clients_important_layers_map)
        self.missing_modalities_by_rounds.append(clients_missing_map)
        
    def ofchp_validate(self):
        self.validate_interface()
        self.max_score_gap = max(self.get_scores_gap, self.max_score_gap)
        self.min_score_gap = min(self.get_scores_gap, self.min_score_gap)
        
    def cluster_validate(self):
        # self.cluster_scores = dict()
        cluster_avg_score = 0
        total_samples = 0
        for cluster in self.clusters:
            # self.cluster_scores[cluster.id] = cluster.validate(self.task)
            cluster.validate(self.task)
            self.slog.info(f'cluster {cluster.id}: {cluster.cluster_score}')
            total_samples += cluster.cluster_samples
            cluster_avg_score += cluster.cluster_score * cluster.cluster_samples
        cluster_avg_score /= total_samples
        self.per_avg_scores.append(cluster_avg_score)
        self.slog.info('avg cluster score:{}'.format(cluster_avg_score))
        
    def attend_clients_validate(self):
        self.cluster_optimal_portfolio = dict()
        for attend_client in self.selected_clients:
            attend_client.validate(self.task)
        self.slog.info('global score gap:',self.get_scores_gap)
        # current_cluster_loss = {}
        for cluster_id,clients in self.cluster_attend_clients.items():
            # self.cluster_optimal_portfolio[cluster_id] = self.clusters[cluster_id].optimal_portfolio(clients)
            self.cluster_optimal_portfolio[cluster_id] = self.clusters[cluster_id].opt_portfolio(clients)
            self.clusters[cluster_id].do_compute_banzhaf(clients, self.get_normalized_score)
            self.slog.info('cluster {} optimal portfolio {}'.format(cluster_id,self.cluster_optimal_portfolio[cluster_id]))
            self.slog.debug('cluster {} loss: {:.3f}'.format(cluster_id,self.clusters[cluster_id].get_training_loss(clients))) 
    
    @property
    def get_scores_gap(self):
        if len(self.global_scores) > 1:
            score_gap = self.global_scores[-1] - self.global_scores[-2]
        else:
            score_gap = self.global_scores[-1]
        return score_gap
    
    @property   
    def get_normalized_score(self):
        return norm.maxmin_norm(self.get_scores_gap,self.max_score_gap,self.min_score_gap)
    
    def get_cluster_probability(self):
        self.cluster_select_probabilities = dict()
        for cluster in self.clusters:
            cluster.compute_prob_powers()
            self.cluster_select_probabilities[cluster.id] = cluster.update_cluster_probability(self.select_prob)
            self.slog.debug('cluster probabilities:{}'.format(self.cluster_select_probabilities[cluster.id]))
    
    def cluster_by_Umasks(self):
        Umasks = []
        cluster_start_time = time.time()
        for client in self.all_clients:
            Umasks.append(client.get_U_mask())
        
        sim_mat = self.calculating_adjacentcy(Umasks)
        saver.save_as_type(
            data = sim_mat,
            file_name='umasks',
            data_path='repository/',
            mode='wb'
        )
        clusters = hierarchical_clustering(copy.deepcopy(sim_mat),self.cluster_alpha,self.linkage)
        self.slog.info('Total cluster time: {}'.format(time.time() - cluster_start_time))
        self.clients_map_clusters = dict()
        self.cluster_map_clients = [[] for _ in range(len(clusters))]
        # self.cluster_ids = [[] for _ in range(len(clusters))]
        for cluster_id,clients_id in enumerate(clusters):
            for client_id in clients_id:
                self.cluster_map_clients[cluster_id].append(self.all_clients[client_id])
                self.clients_map_clusters[self.all_clients[client_id].id] =  cluster_id
        self.cluster_ids = clusters
        
    def calculating_adjacentcy(self,Umasks):
        similarity_matrix = np.zeros([self.num_clients,self.num_clients])
        for c1 in range(self.num_clients):
            mask = np.ones((self.num_clients,), dtype=bool)  
            mask[c1] = False  # 将对角线上的元素设置为False  
            for c2 in range(self.num_clients):
                U1 = copy.deepcopy(Umasks[c1])
                U2 = copy.deepcopy(Umasks[c2])
                mu1 = np.clip(U1.T@U2,a_min = -1.0,a_max = 1.0)
                similarity_matrix[c1,c2] = np.min(np.arccos(mu1)) * 180 / np.pi
            furthest_sid = np.argmax(similarity_matrix[c1][mask]) 
            furthest_sid = furthest_sid+1 if furthest_sid >= c1 else furthest_sid
            closed_sid = np.argmin(similarity_matrix[c1][mask])
            closed_sid = closed_sid+1 if closed_sid >= c1 else closed_sid
            self.slog.debug('{} is furthest away {}, distance: {}'.format(c1,furthest_sid,similarity_matrix[c1,furthest_sid]))
            self.slog.debug('{} is closest to {}, distance: {}'.format(c1,closed_sid,similarity_matrix[c1,closed_sid]))
        return similarity_matrix