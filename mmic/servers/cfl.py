import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from .ditto import Ditto
from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from clients.cfl import ClusterFL as ClusterFLClient
import time
from utils.data import read_client_data
from algorithm.sim.cfl import pairwise_angles
import torch
from utils.data import flatten
from sklearn.cluster import AgglomerativeClustering
from cluster.clusterbase import ClusterBase
EPS_1 = 0.4
EPS_2 = 1.6

class ClusterFL(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.W = {key : value for key, value in self.global_model.named_parameters()}
        self.clusters = []
        self.set_clients(ClusterFLClient)
        self.set_late_clients(ClusterFLClient)
        self.start_to_cluster_rounds = args['cluster']['start_to_cluster']
    
    def evaluate_generalized(self,new_attend_clients=[]):
        for _,c in enumerate(new_attend_clients):
            self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
    
    def train(self):
        new_attend_clients = []
        self.slog.debug('-'*20,'server ClusterFL starting to train','-'*20)
        is_new_joined = False
        for i in range(self.global_rounds+1):
            s_t = time.time()
            if i == 0:
                # make sure all clients have dW
                self.selected_clients = self.all_clients
            else:
                self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            if is_new_joined:
                self.evaluate_generalized(new_attend_clients)
                
            for client in self.selected_clients:
                client.cfl_train()
                if i >= self.start_to_cluster_rounds:  
                    client.reset()

            #limit it temporary in 20 rounds
            if i >= self.start_to_cluster_rounds:
                similarities = self.compute_pairwise_similarities(self.clients)
                # for idc in self.cluster_ids:
                #     if len(idc) <= 2:
                #         continue
                #     max_norm = self.compute_max_update_norm([self.clients[i] for i in idc])
                #     mean_norm = self.compute_mean_update_norm([self.clients[i] for i in idc])
                #     # if the reach the threshold
                #     if mean_norm < EPS_1 and max_norm > EPS_2:
                        # self.cache_model(idc,self.clients[idc[0]].W,acc_clients)
                self.cluster_clients(similarities)
            
            self.receive_models()
            self.cluster_receive_models()
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()
            self.cluster_aggregate_parameters()
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])

            if self.new_clients_settings['enabled'] and i >= self.start_new_joining_round and len(self.late_clients) > 0:
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

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()
    
    def compute_min_distance_with_cluster(self,new_client):
        min_cluster_id = 0
        min_angles = 180
        for cluster in self.clusters:
            angle = self.get_mean_distance(cluster,new_client)
            if angle < min_angles:
                min_angles = angle
                min_cluster_id = cluster.id
        return min_cluster_id
    
    def get_mean_distance(self,cluster,new_client):
        distances = []
        s1 = flatten(new_client.dW)
        for client in cluster.clients:
            s2 = flatten(client.dW)
            distances.append(torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12))
        return torch.mean(torch.stack(distances)).item()
    
    def fine_tuning_new_clients(self,new_clients):
        for new_client in new_clients:
            # We need to get dW firstly
            # self.first_step_fine_tuning(new_client)
            close_cluster_id = self.compute_min_distance_with_cluster(new_client=new_client)
            self.slog.info('cluster close to {} is cluster {}'.format(new_client.id,close_cluster_id))
            
            #add to new client
            self.cluster_map_clients[close_cluster_id].append(new_client)
            self.clients_map_clusters[new_client.id] = close_cluster_id
            self.cluster_ids[close_cluster_id].append(new_client.id)
            
            new_client.set_parameters(self.clusters[int(close_cluster_id)].cluster_model)
            # new_client.model = copy.deepcopy(self.clusters[int(which_cluster)].cluster_model)
            optimizer = torch.optim.SGD(new_client.model.parameters(),lr = self.learning_rate)
            lossFunc = torch.nn.CrossEntropyLoss()
            train_loader = new_client.load_train_data()
            new_client.model.train()
            new_client.copy(target=new_client.PreW,source=new_client.W)
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
            new_client.subtract_(target=new_client.dW, minuend=new_client.W, subtrahend=new_client.PreW)
    
    def compute_pairwise_similarities(self, clients):
        start_time = time.time()
        angles = pairwise_angles([client.dW for client in clients])
        self.slog.info('compute pairwise similarities time:{:2f}'.format(time.time()-start_time))
        return angles
        
    def cluster_by_pairwise_similarities(self,clients):
        return pairwise_angles([client.dW for client in clients])

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()
    
    def cluster_clients(self,S):
        cluster_start_time = time.time()
        cluster_res = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
        self.clients_map_clusters = dict()
        self.cluster_map_clients = [[] for _ in range(cluster_res.n_clusters_)]
        self.cluster_ids = [[] for _ in range(cluster_res.n_clusters_)]
        for client_sid,cluster_id in enumerate(cluster_res.labels_):
            # map{client_id : cluster_id}
            self.clients_map_clusters[self.clients[client_sid].id] = cluster_id
            # map{cluster_id : [clients]}
            self.cluster_map_clients[cluster_id].append(self.clients[client_sid])
            # map{cluster_id : [client_id]}
            self.cluster_ids[cluster_id].append(client_sid)
        # set new clusters object
        self.set_clusters()
        self.slog.info('server,cluster time:{:.3f}s'.format(time.time() - cluster_start_time))
        print('-'*25)
        for i,c in enumerate(self.cluster_ids):
            print('cluster {}:{}'.format(i,c))
            print('-'*25)
        
    def set_clusters(self):
        # clear the clusters
        self.clusters.clear()
        for i in range(len(self.cluster_ids)):
            self.clusters.append(
                ClusterBase(
                    i,
                    self.cluster_ids[i],
                    self.cluster_map_clients[i],
                    None,
                )
            )
    
    def cluster_aggregate_parameters(self):
        if len(self.clusters) <= 0:
            self.slog.info('No clusters in server')
            cluster = [self.clients]
        else:
            for cluster_id in self.cluster_attend_clients.keys():
                self.clusters[cluster_id].aggregate_parameters()
            cluster = self.cluster_map_clients
        self.aggregate_clusterwise(cluster)
        
    
    def aggregate_clusterwise(self, client_clusters):
        for cluster in client_clusters:
            self.reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[client.dW for client in cluster])
    
    def reduce_add_average(self,targets,sources):
        for target in targets:
            for name in target:
                tmp = torch.mean(torch.stack([source[name].data for source in sources]),dim=0).clone()
                target[name].data += tmp