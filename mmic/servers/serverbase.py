import torch
import os
from utils.data import read_client_data
import numpy as np
import h5py
import copy
import time
import const.constants as const
from datetime import datetime
from fedlog.logbooker import Attender
from fedlog.logbooker import ColoredConsoleHandler
import pandas as pd
import utils.time as timeuitls
from const.settings import Global

class Server:
    def __init__(self,args) -> None:
        self.algorithm = args['algorithm']
        if self.algorithm in args['fedAlgorithm'].keys():
            self.fedAlgorithm = args['fedAlgorithm'][self.algorithm]
        # define log
        # 获取当前时间  
        self.set_server_log(args)
        self.args = args
        self.global_model = args['model']
        self.device = args['device']
        self.dataset = args['dataset']
        self.num_clients = args['num_clients']
        self.join_ratio = args['join_ratio']
        self.client_drop_rate = args['client_drop_rate']
        self.learning_rate = args['learning_rate']
        
        self.global_rounds = args['global_rounds']
        self.time_threthold = args['time_threthold']
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.dataset_dir = const.DIR_DEPOSITORY
        if 'dataset_dir' in args.keys():
            self.dataset_dir = args['dataset_dir']
        #join clients setting

        self.new_clients_settings = args['new_clients']
        self.random_clients_selected = args['random_clients_selected']
        self.new_clients_rate = self.new_clients_settings['rate']
        self.set_for_new_clients()
        self.num_original_clients = self.num_clients - self.num_new_clients
        self.num_join_clients = self.num_original_clients * self.join_ratio
        self.num_new_clients = int(self.num_clients * self.new_clients_rate)
        self.late_clients = []
        self.all_clients = []
        self.clients = []
        # false temporary
        self.eval_new_clients = False
        self.fine_tuning_epoch = 0
        self.algorithm = args['algorithm']
        self.eval_gap = args['eval_gap']
        self.budget = []
        
        self.save_dir = 'results'
        if 'save_dir' in args.keys() and args['save_dir'] != '':
            self.save_models_dir = args['save_dir']
            self.save_dir = args['save_dir']

        #cluster
        self.clusters = []
        self.clients_map_clusters = dict()
        self.cluster_map_clients = []
        self.cluster_ids =[]
        # clients info collection
        self.collections = [dict() for _ in range(self.num_clients)]
        Global.total_round = self.global_rounds
        
    def set_for_new_clients(self):
        #whether the new clients join setting is openning, otherwise no clients join in future.
        if not self.new_clients_settings['enabled']:
            self.new_clients_rate = 0
        #total new clients
        self.num_new_clients = int(self.num_clients  * self.new_clients_rate)
        #which round that have new clients to join in
        self.start_new_joining_round = self.new_clients_settings['started_round']
        #how many clients join in each round
        self.num_new_join_each_round = self.new_clients_settings['num_join_each_round']
        # self.late_clients =']
        

    def set_clients(self,clientObj):
        for i in range(self.num_original_clients):
            train_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=True)
            test_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=False)
            client = clientObj(
                self.args,
                id = const.ORIGINAL+str(i),
                serial_id=i,
                train_samples = len(train_data),
                test_samples = len(test_data),
                logkey = self.logkey,
            )
            self.clients.append(client)
        self.all_clients.extend(self.clients)
        
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
    
    def send_models(self):
        assert (len(self.clients) > 0)
        
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time['rounds'] += 1
            client.send_time['total_cost'] += 2 * (time.time() - start_time)
    
    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)
    
    def add_parameters(self,w,client_model):
        for server_param,client_param in zip(self.global_model.parameters(),client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    
    def save_global_model(self):
        model_path = os.path.join(self.save_models_dir,self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,self.algorithm + '_server' + '.pt')
        print('server saving directory:',model_path)
        torch.save(self.global_model,model_path)
    
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = os.path.join("results",algo)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + datetime.now().strftime(const.TIME_FORMAT)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        self.slog.info('rs_test_acc', self.rs_test_acc,'rs_test_auc', self.rs_test_auc,'rs_train_loss',self.rs_train_loss)
    
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            # if c.id.startswith('late') and c.id in self.clients_map_clusters.keys():
            #     self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
            ct,ns,auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        
        return ids,num_samples,tot_correct,tot_auc
    
    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0],[1],[0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl,ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        
        ids = [c.id for c in self.clients]
        return ids,num_samples,losses
    
    def evaluate(self,acc=None,loss=None):
        test_metrics_res = self.test_metrics()
        train_metrics_res = self.train_metrics()
        
        test_acc = sum(test_metrics_res[2]) * 1.0 / sum(test_metrics_res[1])
        test_auc = sum(test_metrics_res[3]) * 1.0 / sum(test_metrics_res[1])
        
        train_loss = sum(train_metrics_res[2]) * 1.0 / sum(train_metrics_res[1])
        accuracies = [correct / num for correct,num in zip(test_metrics_res[2],test_metrics_res[1])]
        #about auc, reference:https://zhuanlan.zhihu.com/p/569006692
        auc_collections = [acc / num for acc,num in zip(test_metrics_res[3],test_metrics_res[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        self.slog.info('server: avg test accuracy:{:.3f}'.format(test_acc))
        self.slog.info('server: avg test AUC:{:.3f}'.format(test_auc))
        
        self.slog.info('std: test accuracy:{:.3f}'.format(np.std(accuracies)))
        self.slog.info('std test AUC:{:.3f}'.format(np.std(auc_collections)))
    
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
            self.uploaded_models.append(client.model)
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
    
    def train(self):
        self.slog.debug('server','starting to train')
        for i in range(Global.total_round+1):
            Global.current_round += 1
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))
        
        self.save_running_process()
        self.save_results()
        self.save_global_model()
    
    def save_running_process(self):
        collections = {c.id:[] for c in self.all_clients}
        # for c in self.all_clients:
        #     collections[c.id].extend(
        #         [c.train_samples,c.test_samples]
        #     )
        index = []
        for i in range(0,self.global_rounds):
            index.extend([str(i) + 'train_loss','train_samples','test_accuracy','test_samples'])
            for c in self.all_clients:
                if len(c.collection[i]) > 0:
                    collections[c.id].extend(c.collection[i])
                else:
                    collections[c.id].extend(['not participated'] * 4)
        for c in self.all_clients:
            collections[c.id].extend([c.train_samples,c.test_samples]) 
        index.extend(['total_train_samples','total_test_sampels'])
        # collections['metrics'] = 
        df = pd.DataFrame(collections)
        df.index = index
        df.to_csv(os.path.join(self.save_dir,self.algorithm,timeuitls.getTime()+const.CSV_SUFFIX))
        
    def set_late_clients(self,clientObj):
        for i in range(self.num_new_clients):
            train_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=True)
            test_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=False)
            client = clientObj(
                self.args,id = 'late_'+str(i),
                train_samples = len(train_data),
                test_samples = len(test_data),
                serial_id = i+self.num_original_clients,
                logkey = self.logkey,
            )
            self.late_clients.append(client)
        self.all_clients.extend(self.late_clients)

    def cluster_aggregate_parameters(self):
        # for cluster in self.clusters:
        #     cluster.aggregate_parameters()
        if len(self.clusters) <= 0:
            self.slog.info('No clusters in server')
            return
        for cluster_id in self.cluster_attend_clients.keys():
            self.clusters[cluster_id].aggregate_parameters()
            
    def evaluate_generalized(self,new_attend_clients=[]):
        for _,c in enumerate(new_attend_clients):
            self.clusters[self.clients_map_clusters[c.id]].test_model_generalized(c)
    
    def set_server_log(self,args):
        self.logkey = args['dataset']+'_'+datetime.now().strftime(const.LOG_DIR_TIME_FORMAT)
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():   
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir,self.algorithm,self.logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.slog = Attender(
            index=const.SERVER_KEY,
            filePath=os.path.join(log_path,const.SERVER_KEY+const.LOG_SUFFIX),
            handlers=[ColoredConsoleHandler(bubble=True,format_string=const.LOG_FILE_FORMAT)],
        )
    
    def cluster_receive_models(self):
        if len(self.clusters) <= 0:
            return
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
