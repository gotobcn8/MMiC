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
from .serverbase import Server
from clients.fedopt import FedOptCient
import numpy as np
import copy

class FedOptServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.set_clients(FedOptCient)
        self.global_optimizer = self._initialize_global_optimizer()
    
    def _initialize_global_optimizer(self):
        # global optimizer
        global_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=self.args['global_learning_rate'],
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

            self.receive_models()
            self.aggregate_parameters()
            # self.check_global_model()
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            
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

    def aggregate_parameters(self):
        #FedOpt global step
        global_model_state = copy.deepcopy(self.uploaded_models[0])
        for key in global_model_state.keys():
            global_model_state[key] = self.uploaded_models[0][key]*self.uploaded_weights[0]
            for i in range(0,len(self.uploaded_models)):
                global_model_state[key] += self.uploaded_models[i][key] * self.uploaded_weights[i]

        # self.global_optimizer = self._initialize_global_optimizer()
        # self.global_optimizer.load_state_dict(global_optimizer_state)
        # self.global_optimizer.step()
        self.fedopt_update(global_model_state)


    def fedopt_update(
        self, 
        update_weights
    ):
        # zero_grad
        self.global_optimizer.zero_grad()
        global_optimizer_state = self.global_optimizer.state_dict()

        # new_model
        new_model = copy.deepcopy(self.global_model)
        new_model.load_state_dict(update_weights, strict=True)

        # set global_model gradient
        with torch.no_grad():
            for param, new_param in zip(
                self.global_model.parameters(), new_model.parameters()
            ):
                param.grad = (param.data - new_param.data) / self.learning_rate

        # replace some non-parameters's state dict
        state_dict = self.global_model.state_dict()
        for name in dict(self.global_model.named_parameters()).keys():
            update_weights[name] = state_dict[name]
        self.global_model.load_state_dict(update_weights, strict=True)

        # optimization
        self.global_optimizer = self._initialize_global_optimizer()
        self.global_optimizer.load_state_dict(global_optimizer_state)
        self.global_optimizer.step()