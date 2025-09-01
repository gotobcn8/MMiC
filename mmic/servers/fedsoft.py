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
from clients.fedsoft import FedSoftClient
from cluster.fedsoft import FedSoftCluster
import const.constants as const

class FedSoftServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.fine_tuning_epoch = args['fine_tuning_epoch']
        self.clients_ids_map = dict()
        # set to client late and original
        self.set_clients(FedSoftClient)
        self.set_late_clients(FedSoftClient)
        self.num_clusters = args['cluster']['cluster_num']
        self.importance_weights_matrix = []
        self._zero_weights = None
        
        self.set_cluster(FedSoftCluster)
        # estimation gap
        self.estimation_interval = self.fedAlgorithm['estimation_interval']
        
    def set_cluster(self,clusterObj):
        self.clusters = []
        for i in range(self.num_clusters):
            self.clusters.append(clusterObj(
                i,
                self.args['model'],
            ))
                
    def train(self):
        new_attend_clients = []
        self.slog.debug('server','starting to train')
        is_new_joined = False
        for i in range(self.global_rounds+1):
            s_t = time.time()
            
            if i % self.estimation_interval == 0 or is_new_joined:
                self.importance_weights_matrix = []
                for client in self.clients:
                    client.estimate_importance_weights(self.clusters)
                    self.importance_weights_matrix.append(client.get_importance())
                self.importance_weights_matrix = np.array(self.importance_weights_matrix)
                self.importance_weights_matrix /= np.sum(self.importance_weights_matrix,axis=0)
            
            self.selected_clients = self.select_clients()
            # self.send_models()
            
            if i%self.eval_gap == 0:
                self.slog.debug(f"-------------Round number: {i}-------------")
                self.slog.debug("start evaluating model")
                self.evaluate()
            if is_new_joined:
                # If there are some new clients join, need to test the generalization firstly.
                self.evaluate_generalized(new_attend_clients)
                
            for client in self.selected_clients:
                client.train()

            self.cluster_aggregate()
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
    
    def fine_tuning_new_clients(self,new_clients):
        for new_client in new_clients:
            new_client.estimate_importance_weights(self.clusters)
            cluster_id = np.argmax(new_client.importance_estimated)
            new_client.set_parameters(self.clusters[int(cluster_id)].cluster_model)
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
            cluster_id = np.argmax(a= self.importance_weights_matrix[c.serial_id,])
            selected_test_clients = np.random.choice(
                a = self.clients,
                size = int(len(self.clients) * const.TEST_GENERALIZATION_RATE),
                replace = False,
                p = self.importance_weights_matrix[:, cluster_id]
            )
            self.clusters[cluster_id].test_model_generalized(selected_test_clients,c)
    
    def generate_zero_weights(self):
        if self._zero_weights is None:
            self._zero_weights = {}
            for key, val in self.global_model.state_dict().items():
                self._zero_weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)
        return copy.deepcopy(self._zero_weights)

    def cluster_aggregate(self):
        for cluster in self.clusters:
            nextweights = self.generate_zero_weights()
            for selected_client in self.selected_clients:
                aggregated_weight = 1.0 / self.current_num_join_clients
                client_weights = selected_client.get_model_dict()
                for key in nextweights.keys():
                    nextweights[key] += aggregated_weight * client_weights[key].cpu()
            cluster.cluster_model.load_state_dict(state_dict = nextweights)
            
    def select_clients(self):
        selected_clients = []
        self.current_num_join_clients = len(self.clients) * (1 - self.client_drop_rate) * self.join_ratio
        cset = set()
        for cluster in self.clusters:
            temp_selected = np.random.choice(
                    a = self.clients,
                    size = int(self.current_num_join_clients),
                    replace = False,
                    p = self.importance_weights_matrix[:, cluster.id]
                ).tolist()
            for temp_c in temp_selected:
                if temp_c.serial_id not in cset:
                    selected_clients.append(temp_c)
                    cset.add(temp_c.serial_id)
        return selected_clients