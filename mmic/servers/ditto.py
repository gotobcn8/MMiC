import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from threading import Thread
import os
import torch
import torch.nn.functional as F
import math
import utils.dlg as dlg

class Ditto(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        
        self.set_clients(ClientDitto)
        ditto = args['fedAlgorithm']['Ditto']
        self.dlg_gap = ditto['dlg_gap']
        self.dlg_eval = ditto['dlg_eval']
        
        self.slog.info(f"join ratio total clients:{self.join_ratio / self.num_clients}")
        self.slog.info('Finished creating server and clients')
    
    def train(self):
        for i in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            
            if i % self.eval_gap == 0:
                self.slog.debug(f"-------------Round number: {i}-------------")
                self.slog.debug("start evaluating model")
                self.evaluate()
                self.slog.debug('Evaluating personalized models')
                self.evaluate_personalized()

            #select different clients each round
            for client in self.selected_clients:
                #personalized model train
                client.train_personalized()
                #based on global model train
                client.train()
                
            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()
            self.budget.append(time.time() - start_time)
            self.slog.info('-'*20,f'round {i} training time:{self.budget[-1]}','-'*20)
        self.slog.info('Best Accuracy: {:.2f}'.format(max(self.rs_test_acc)))

        self.slog.info(f'Average time cost per round:{sum(self.budget[1:])/len(self.budget[1:])}')
        self.save_global_model()

    def train_metrics(self):
        return self.train_metrics_personalized()
    
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
    
    def test_metrics_personalized(self):
        # return super().test_metrics()
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        num_samples = []
        total_corrects = []
        total_auc = []
        ids = [0] * len(self.clients)
        for i,c in enumerate(self.clients):
            c_corrects,c_num_samples,c_auc = c.test_metrics()
            total_corrects.append(c_corrects*1.0)
            total_auc.append(c_auc * c_num_samples)
            num_samples.append(c_num_samples)
            ids[i] = c.id
        
        return ids,num_samples,total_corrects,total_auc 
    
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
    
    def call_dlg(self,id):
        cnt = 0
        psnr_val = 0
        for cid,client_model in zip(self.uploaded_ids,self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(),client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            
            target_inputs = []
            train_loader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i,(x,y) in enumerate(train_loader):
                    x = x.to(self.device)
                    # y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x,output))
            
            d = dlg.DLG(client_model,origin_grad,target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
        if cnt > 0:
            self.slog.info('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            self.slog.error('PSNR Error')
        