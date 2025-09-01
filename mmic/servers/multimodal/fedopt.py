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
from .multimodalserver import Server
from clients.multimodal.fedopt import MMFedOptCient
import numpy as np
import copy

class MMFedOptServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.set_clients(MMFedOptCient)
        self.global_learning_rate = self.args.get('global_learning_rate',1e-2)
        self.beta1 = self.args.get('beta1',0.9)
        self.beta2 = self.args.get('beta',0.999)
        self.tau = self.args.get('tau',1e-3)
        self.global_optimizer = self._initialize_global_optimizer()
        self.__init_momentums_velocities()
        self.delta_list: list[torch.Tensor] = None
        # self.optimize_type = 'adagrad'
        self.optimize_type = 'adagrad'
        self.update_maps = {
            "adagrad": self._update_adagrad,
            "yogi": self._update_yogi,
            "adam": self._update_adam,
        }[self.optimize_type]
    
    def __init_momentums_velocities(self):
        self.momentums = {}
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            self.momentums[key] = torch.zeros_like(global_state_dict[key])
        # for key,param in self.global_model.named_parameters():
        #     self.momentums[key] = torch.zeros_like(param.data)
        self.velocities = copy.deepcopy(self.momentums)
      
    def _initialize_global_optimizer(self):
        # global optimizer
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr = self.global_learning_rate,
            momentum=0.9,
            weight_decay=0.0
        )
        return global_optimizer
    
    def train(self):
        self.slog.debug('server','starting to train')
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
 
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.validate_interface()
            for client in self.selected_clients:
                client.train()
            if i % self.eval_gap == 0:
                self.attend_clients_validate()
                
            self.receive_models()
            self.aggregate_parameters()
            # self.check_global_model()
            
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            self.tracker.clear_cache()
        self.save_collector()
        print("\nBest accuracy.")
        print(f'max:{max(self.server_test_acc)}\n{self.server_test_acc}')
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))
        
        # self.save_running_process()
        self.save_results()
        self.save_global_model()
    # def evalate():
    #     pass
    
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

    def receive_models(self):
        if len(self.selected_clients) <= 0:
            self.slog.exception("selected clients couldn't 0")
        
        self.uploaded_cids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in self.selected_clients:
            try:
                avg_train_time_cost = client.train_time['total_cost'] / client.train_time['rounds']
                avg_send_time_cost = client.send_time['total_cost'] / client.send_time['rounds']
                client_time_cost = avg_train_time_cost + avg_send_time_cost
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost > self.time_threthold:
                continue
            total_samples += client.train_samples
            self.uploaded_cids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.state_dict())
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
            
        self.get_params_diff()
        
    def get_params_diff(self):
        self.params_diff = []
        global_state_dict = self.global_model.state_dict()
        for state_dict in self.uploaded_models:
            diffdict = {}
            for key in state_dict:
                diffdict[key] = global_state_dict[key] - state_dict[key]
            self.params_diff.append(diffdict)
    
    def aggregate_parameters(self):
        #FedOpt global step
        # global_model_state = 
        self.old_global_model = copy.deepcopy(self.global_model)
        global_model_state = self.global_model.state_dict()
        # global_model_state = copy.deepcopy(self.uploaded_models[0])
        self.fedopt_update_recent(global_model_state)

    @torch.no_grad()
    def fedopt_update_recent(self,global_model_state):
        weighted_params_diff = {}
        for i in range(len(self.selected_clients)):
            # for key,_ in self.global_model.named_parameters():
            for key in global_model_state:
                weighted_params_diff[key] = weighted_params_diff.get(key,0) + (self.uploaded_weights[i] * self.params_diff[i][key])
        
        for key in self.momentums:
            self.momentums[key] = self.beta1 * self.momentums[key] + (1 - self.beta1) * weighted_params_diff[key]
        
        self.update_maps(weighted_params_diff)
        
        for key in self.momentums:
            global_model_state[key] = global_model_state[key] - self.global_learning_rate * (self.momentums[key] / (self.velocities[key].sqrt() + self.tau))
        
        self.global_model.load_state_dict(global_model_state, strict=True)
    
    def _update_adagrad(self, delta_list):
        for key in self.velocities:
            self.velocities[key] = self.velocities[key] + delta_list[key] ** 2

    def _update_yogi(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            delta_pow2 = delta ** 2
            v.data = v - (1 - self.beta2) * delta_pow2 * torch.sign(v - delta_pow2)

    def _update_adam(self, delta_list):
        for key in self.velocities:
            self.velocities[key] = self.beta2 * self.velocities[key] + (1 - self.beta2) * delta_list[key] ** 2
