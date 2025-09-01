import torch
import torch.nn as nn
import numpy as np
from .client import ClientBase
from models.optimizer.fedoptimizer import SCAFFOLDOptimizer
import time
import copy

class ClientSCAFFOLD(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,**kwargs):
        super().__init__(args,id,train_samples,test_samples,**kwargs)

        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5, weight_decay=0)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=self.learning_rate_decay_gamma
        )   
        # self.learning_rate_decay = True
        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None
    
    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)
                # self.optimizer.step()
                # model_params = self.model.state_dict()
                # for p,sc,cc in zip(self.model.parameters(),self.global_c,self.client_c):
                #     # for p,sc,cc in zip(group['params'],):
                #     p = p - (self.learning_rate*(sc-cc))
        # self.model.cpu()
        self.num_batches = len(trainloader)
        self.update_yc()
        # self.delta_c, self.delta_y = self.delta_yc()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time['rounds'] += 1
        self.train_time['total_cost'] += time.time() - start_time
            
        
    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self):
        if self.num_batches <= 0:
            return
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches/self.learning_rate * (x - yi)

    def delta_yc(self):
        delta_y = []
        delta_c = []
        # if self.modele.parameters is
        if self.num_batches <= 0:
            for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
                delta_y.append(yi - x)
                delta_c.append(-c + 1/self.num_batches/self.learning_rate * (x - yi))
        return delta_y, delta_c
