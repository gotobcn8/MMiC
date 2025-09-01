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
import torch
from utils.data import flatten
from sklearn.cluster import AgglomerativeClustering
from cluster.clusterbase import ClusterBase
from models.init import weight_init

class IFCAServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.clusters = []
        self.set_clients(ClusterFLClient)
        self.set_late_clients(ClusterFLClient)
        self.num_clusters = args['cluster']['cluster_num']
        self.set_clusters()
        self.update_clusters()
        self.cluster_time = 0.0
    
    def set_clusters(self):
        for i in range(self.num_clusters):
            model = copy.deepcopy(self.global_model)
            model.apply(weight_init)
            self.clusters.append(ClusterBase(
                id=i,
                cluster_ids=[],
                clients=[],
                model=model,
            ))
    
    def update_clusters_samples(self):
        for cluster in self.clusters:
            cluster.get_cluster_sampels()
    
    def train(self):
        is_new_joined = False
        new_attend_clients = []
        self.slog.debug('-'*20,'server ClusterFL starting to train','-'*20)
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            if is_new_joined:
                self.evaluate_generalized(new_attend_clients)
            
            self.update_clusters()
            for client in self.selected_clients:
                self.ifca_train(client)
            
            self.update_clusters_samples()
            
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
                for client in new_attend_clients:
                    self.ifca_train(client)
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
        print('total cluster time:{:.4f}s'.format(self.cluster_time))
        self.save_results()
        self.save_global_model()
    
    def update_clusters(self):
        for c in self.clusters:
            c.clients.clear()
            c.cluster_ids.clear()
        self.clients_map_clusters = dict()
        self.clusters_ids = [[] for _ in range(self.num_clusters)]
        self.clusters_map_clients = [[] for _ in range(self.num_clusters)]    

    def ifca_train(self,client):
        test_acc_collections = []
        start_time = time.time()
        for cluster in self.clusters:
            test_acc,_,_ = client.test_other_model(cluster.cluster_model)
            test_acc_collections.append(test_acc)
        best_cluster_id = np.argmax(test_acc_collections)
        finished_time = time.time() - start_time
        self.cluster_time += finished_time
        self.slog.info('cluster time:{:.3f}s'.format(finished_time))
        # this is the most suitable cluster 
        self.clusters[best_cluster_id].clients.append(client)
        self.clusters[best_cluster_id].cluster_ids.append(client.id)
        # update basic info of cluster
        self.clusters_ids[best_cluster_id].append(client.id)
        self.clusters_map_clients[best_cluster_id].append(client)
        self.clients_map_clusters[client.id] = best_cluster_id
        client.model = copy.deepcopy(self.clusters[best_cluster_id].cluster_model)
        client.train()