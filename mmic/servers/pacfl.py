import copy
import random
import time
import torch
from servers.serverbase import Server
import numpy as np
from clients.pacfl import PACFL as PACFLClient
from algorithm.sim.cluster import hierarchical_clustering
from cluster.clusterbase import ClusterBase

class PACFL(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.set_for_pacfl_clients(PACFLClient)
        self.cluster_alpha = self.fedAlgorithm['alpha']
        self.linkage = self.fedAlgorithm['linkage']
        
    def set_for_pacfl_clients(self,clientObj):
        self.set_clients(clientObj)
        self.set_late_clients(clientObj)
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        self.slog.info('total calculating time {:.3f}s'.format(time.time() - start_time))
    
    def evaluate_generalized(self,new_attend_clients=[]):
        for _,c in enumerate(new_attend_clients):
            self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
    
    def train(self):
        self.cluster_by_Umasks()
        self.set_clusters()
        self.slog.debug(self.algorithm,'Starting to train')
        is_new_joined = False
        new_attend_clients = []
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients(new_attend_clients)
            self.send_models()
            
            if i%self.eval_gap == 0:
                self.slog.debug(f"-------------Round number: {i}-------------")
                self.slog.debug("start evaluating model")
                self.evaluate()
            #whether there are late clients still existed.
            if is_new_joined:
                # If there are some new clients join, need to test the generalization firstly.
                self.evaluate_generalized(new_attend_clients)
                
            for client in self.selected_clients:
                client.train()

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

        self.save_results()
        self.save_global_model()
        return

    def set_clusters(self):
        self.clusters = []
        for id,cluster_clients in enumerate(self.cluster_map_clients):
            self.clusters.append(
                ClusterBase(
                    id = id,
                    cluster_ids = self.cluster_ids[id],
                    clients=cluster_clients,
                    model = self.args['model'],
                )
            )

    def cluster_by_Umasks(self):
        Umasks = []
        cluster_start_time = time.time()
        for client in self.all_clients:
            Umasks.append(client.get_U_mask())
        sim_mat = self.calculating_adjacentcy(Umasks)
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