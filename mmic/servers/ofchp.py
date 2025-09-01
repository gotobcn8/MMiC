import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from .ditto import Ditto
from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from models.optimizer.ditto import PersonalizedGradientDescent
from clients.ofchp import OFCHPClient
import time
from utils.data import read_client_data
from sklearn.cluster import KMeans
from cluster.clusterbase import ClusterBase
import torch
import const.constants as const
from algorithm.augmentation import augmentation
from const.settings import Global

class OFCHPServer(Server):
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
        self.set_for_lsh_clients(OFCHPClient)
        self.num_clusters = args['cluster']['cluster_num']
        self.pre_cluster(self.num_clusters)
        self.set_clusters(ClusterBase,args)

    def set_clusters(self,clusterObj,args):
        self.clusters = []
        for i,cluster in enumerate(self.cluster_ids):
            self.clusters.append(clusterObj(i,cluster,self.cluster_map_clients[i],args['model']))

    def set_clients(self,clientObj):
        for i in range(self.num_original_clients):
            train_samples = len(read_client_data(self.dataset,i,self.dataset_dir,is_train=True))
            test_samples = len(read_client_data(self.dataset,i,self.dataset_dir,is_train=False))
            if self.augmentation is not None and train_samples < self.data_volume:
                self.slog.info('client {} samples is {}, doing augmentation'.format(i,train_samples))
                train_samples = augmentation.DoAugmentation(self.dataset,self.dataset_dir,i,self.augmentation)
            client = clientObj(
                self.args,
                id = const.ORIGINAL+str(i),
                serial_id=i,
                train_samples = train_samples,
                test_samples = test_samples,
                logkey = self.logkey,
            )
            self.clients.append(client)
        self.all_clients.extend(self.clients)

    def set_for_lsh_clients(self,clientObj):
        self.set_clients(clientObj)
        self.set_late_clients(clientObj)
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        for i,client in enumerate(self.all_clients):
            client.count_sketch(self.hashF)
            #  = self.hashF.hash(client)
            self.sketches[client.id] = client.minisketch
            self.clients_ids_map[client.id] = i
        self.slog.info('total calculating time {:.3f}s'.format(time.time() - start_time))

    

    def select_clients(self,new_attend_clients):
        self.slog.info('Starting select clients for server')

        if self.random_clients_selected:
            #random number of attend clients
            self.current_num_join_clients = np.random.choice(int(self.num_original_clients * self.join_ratio),self.num_original_clients+1)
        else:
            #static number of attend clients
            self.current_num_join_clients = self.num_join_clients
        #That guarantee the late clients can be selected first time.
        selected_clients = list(np.random.choice(self.clients,int(self.current_num_join_clients)-len(new_attend_clients),replace=False))
        selected_clients.extend(new_attend_clients)
        return selected_clients
                
    def train(self):
        new_attend_clients = []
        self.slog.debug('server','starting to train')
        is_new_joined = False
        for i in range(Global.total_round+1):
            Global.current_round += 1
            s_t = time.time()
            self.selected_clients = self.select_clients(new_attend_clients)
            self.send_models()
            
            if i%self.eval_gap == 0:
                self.slog.debug(f"-------------Round number: {i}-------------")
                self.slog.debug("start evaluating model")
                self.evaluate()
                self.slog.debug('Evaluating personalized models')
                self.evaluate_personalized()
            #whether there are late clients still existed.
            if is_new_joined:
                # If there are some new clients join, need to test the generalization firstly.
                self.evaluate_generalized(new_attend_clients)
                
            for client in self.selected_clients:
                client.train()
                client.train_personalized()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.cluster_receive_models()
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()
            self.cluster_aggregate_parameters()
            
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])

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
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))

        self.save_running_process()
        self.save_results()
        self.save_global_model()
    
    def cluster_receive_models(self):
        self.cluster_attend_clients = dict()
        for attend_client in self.selected_clients:
            cluster_id = self.clients_map_clusters[attend_client.id]
            if cluster_id not in self.cluster_attend_clients:
                self.cluster_attend_clients[cluster_id] = []
            #attender serial id in this round
            self.cluster_attend_clients[cluster_id].append(attend_client)
        
        for cluster in self.clusters:
            if cluster.id in self.cluster_attend_clients.keys():
                cluster.receive_models(self.cluster_attend_clients[cluster.id])
    
    def pre_cluster(self,cluster_num):
        sketches1dim = []
        cluster_start_time = time.time()
        for client in self.all_clients:
            sketch = client.minisketch
            print(np.count_nonzero(sketch))
            sketches1dim.append(sketch.reshape(1,-1)[0]) 
            # print(sketch.size())
        kmeans = KMeans(n_clusters=cluster_num)
        #fit the kmeans with the one-dimensional data to cluster
        kmeans_res = kmeans.fit(sketches1dim)
        self.slog.info('server,cluster time:{:.3f}s'.format(time.time() - cluster_start_time))
        self.clients_map_clusters = dict()
        '''
        from the cluster form a map to the {client_id:cluster_id}
        from the cluster form a map to the {cluster id: [clients_obj]}
        '''
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
    
    def evaluate_personalized(self,acc=None,loss=None):
        test_metrics_res = self.test_metrics_personalized()
        train_metrics_res = self.train_metrics_personalized()
        
        test_acc = sum(test_metrics_res[2]) * 1.0 / sum(test_metrics_res[1])
        test_auc = sum(test_metrics_res[3]) * 1.0 / sum(test_metrics_res[1])
        
        train_loss = sum(train_metrics_res[2]) * 1.0 / sum(train_metrics_res[1])
        accuracies = [correct / num for correct,num in zip(test_metrics_res[2],test_metrics_res[1])]
        #about auc, reference:https://zhuanlan.zhihu.com/p/569006692
        auc_collections = [acc / num for acc,num in zip(test_metrics_res[3],test_metrics_res[1])]
        
        if accuracies == None:
            self.rs_test_acc.append(test_acc)
        else:
            accuracies.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        self.slog.info('server: avg test accuracy:{:.3f}'.format(test_acc))
        self.slog.info('server: avg test AUC:{:.3f}'.format(test_auc))
        
        self.slog.info('std: test accuracy:{:.3f}'.format(np.std(accuracies)))
        self.slog.info('std test AUC:{:.3f}'.format(np.std(auc_collections)))
    
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