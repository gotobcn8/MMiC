from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from algorithm.fab import compute_model_l2_norm
from models.optimizer.ditto import PersonalizedGradientDescent
from clients.ofchp import OFCHPClient
import time
from utils.data import read_client_data
from sklearn.cluster import KMeans
from cluster.clusterbase import ClusterBase
import torch
import const.constants as const
from algorithm.augmentation import augmentation
from .multimodalserver import Server
from clients.multimodal.fedavg import MMFedAvg
import numpy as np
from const.settings import Global
import json
class FedAvg(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.set_clients(MMFedAvg)
    
    def train(self):
        self.slog.debug('server','starting to train')
        clients_norm2_array = []
        global_model_norm2 = []
        for i in range(Global.total_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
 
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.validate_interface()

            for client in self.selected_clients:
                client.train()

            
            
            # Do the personalized client check
            if i % self.eval_gap == 0:
                self.attend_clients_validate()

            self.receive_models()
            
            self.get_clientmodel_norm2(clients_norm2_array)
            
            self.aggregate_parameters()
            
            global_model_norm2.append(compute_model_l2_norm(self.global_model))
            
            # self.check_global_model()
            self.budget.append(time.time() - s_t)
            print(f'avg client model norm2:{clients_norm2_array[-1]:.4f}, global model norm2:{global_model_norm2[-1]:.4f}')
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            self.writer.add_scalar('budget',self.budget[-1],Global.current_round)
        self.save_collector()
        print("\nBest accuracy.")
        print(f'max:{max(self.server_test_acc)}\n{self.server_test_acc}')
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))
        # print('client norm2:',clients_norm2_array,'global model norm2:', global_model_norm2)
        # self.save_running_process()
        self.save_norm2(clients_norm2_array, global_model_norm2)
        self.save_results()
        self.save_global_model()
    # def evalate():
    #     pass
    
    def save_norm2(self, clients_norm2_array, global_model_norm2):
        norm2_content = {
            'client_norm2':clients_norm2_array,
            'global_norm2':global_model_norm2
        }
        with open('norm2.json','w') as f:
            json.dump(norm2_content,f) 
        
    def select_clients(self,is_late_attended = False):
        self.slog.info('Starting select clients for server')
        if self.random_clients_selected:
            #random number of attend clients
            self.current_num_join_clients = np.random.choice(int(self.num_original_clients * self.join_ratio),self.num_original_clients+1)
        else:
            #static number of attend clients
            self.current_num_join_clients = len(self.clients) * (1 - self.client_drop_rate) * self.join_ratio
        selected_clients = list(np.random.choice(self.clients,int(self.current_num_join_clients),replace=False))
        return selected_clients
    
    def get_clientmodel_norm2(self, total_rounds_clients_norm2):
        clients_norm2 = 0
        for i,client in enumerate(self.selected_clients):
            clients_norm2 += self.uploaded_weights[i] * compute_model_l2_norm(client.model)
            
        # clients_norm2 /= len(self.selected_clients)
        total_rounds_clients_norm2.append(clients_norm2)
