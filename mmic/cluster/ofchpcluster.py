import numpy as np
from .clusterbase import ClusterBase
import const.constants as const
from torch.utils.data import DataLoader,SubsetRandomSampler
from utils.collate import collate_mm_fn_padd
import torch
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from algorithm.normalized import get_normalized_value
from algorithm import normalized
from fedlog.logbooker import Attender
import const.constants as const
import os
import queue
import const.tasks as tasks
from utils.multimodaldata import api as mmapi
from datetime import datetime
from fedlog.logbooker import ColoredConsoleHandler
from evaluator.api import DatasetEvaluator
from const.settings import Global
from collections import OrderedDict
class OFCHPCluster(ClusterBase):
    def __init__(self,args, id, cluster_ids, clients) -> None:
        super().__init__(args,id,cluster_ids, clients)
        self.importance_count = dict()
        self.clients_select_times = dict()
        if 'batch_size' not in args:
            self.batch_size = args[args['dataset']]['batch_size']
        else:
            self.batch_size = args['batch_size']
        self.dataset_dir = const.DIR_DEPOSITORY
        if 'dataset_dir' in args.keys():
            self.dataset_dir = args['dataset_dir']
            
        self.dataset = args['dataset']
        self.num_classes = args[args['dataset']].get('num_classes',0)
        self.is_data_preload = args[args['dataset']].get('is_preload',True)
        self.device = args['device']
        self.set_cluster_log(args)
        evaluator = DatasetEvaluator.get(self.dataset)
        if evaluator is not None:
            self.evaluator =  evaluator(
                logger = self.ctrlog,
                logkey = self.logkey,
            )
        self.__init_sid2clients()
        self.__init_risk_clients_storer()
    
    def __init_sid2clients(self):
        self.sid2clients = {}
        
        for i,csid in enumerate(self.cluster_ids):
            self.sid2clients[csid] = self.clients[i]
    
    def __init_risk_clients_storer(self):
        self.risk_storer = {
            self.clients[i].serial_id:dict() for i in range(len(self.clients))
        }
        for i in range(len(self.clients)):
            for j in range(i+1,len(self.clients)):
                ci,cj = min(self.clients[i].serial_id,self.clients[j].serial_id), max(self.clients[i].serial_id,self.clients[j].serial_id)
                # a,b = sorted([self.clients[i].serial_id,self.clients[j].serial_id])
                self.risk_storer[ci][cj] = queue.Queue(maxsize = 3)
    
    def init_clients_selection(self,rate):
        select_rates = {}
        for client in self.clients:
            select_rates[client.sid] = rate
        
    def load_dataset(self,data_type = const.PREFIX_TRAIN,shuffle = True, batch_size=10,index = 'cluster'):
        # if batch_size == None:
        if self.batch_size is not None:
            batch_size = self.batch_size
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        collate_fn = mmapi.CollateGenerator[self.dataset]()
        drop_last = (data_type != const.PREFIX_TEST)
        random_indices = np.random.choice(len(train_data), int(self.cluster_test_subrate * len(train_data)), replace=False)
        sampler = SubsetRandomSampler(random_indices)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=drop_last, shuffle=shuffle,collate_fn=collate_fn,sampler=sampler)
    
    def validate(self,task):
        self.cluster_model.eval()
        if task == tasks.TASK_CLASSIFY:
            self.norm_validate()
        elif task == tasks.TASK_RETRIEVAL:
            self.retrieval_validate()
    
    def retrieval_validate(self):
        test_loader = self.load_dataset(const.DataType[1],shuffle=False,index='cluster')
        res = self.evaluator.evalrank(self.cluster_model,test_loader)
        self.cluster_score = res['rsum']
        
    def norm_validate(self):
        # self.cluster_clients_validate()
        test_metrics_res = self.evaluate()
        test_acc = test_metrics_res[0] * 1.0 / test_metrics_res[2]
        test_score = test_metrics_res[1]
        
        # self.server_test_acc.append(test_acc)
        # self.server_test_auc.append(test_auc)
        # self.cluster_score = {
        #     'acc':test_acc * 100,
        #     'micro_f1': test_score[1] * 100,
        # }
        self.cluster_score = test_score[1] * 100
        # self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        # print(('cluster {}: accuracy: {:.3f}, micro_f1: {:.2f}'.format(self.id,test_acc*100,test_score[1])))
        self.ctrlog.info('cluster {}: accuracy: {:.3f}, micro_f1: {:.2f}'.format(self.id,test_acc*100,test_score[1]))
        # self.slog.info('server: avg test auc:{:.3f}, micro_f1{} ,macro_f1:{}, weighted_f1:{}'.format(test_score[0],test_score[1],test_score[2],test_score[3]))
        # return round(test_score[1]*100,3)
        # self.slog.info('std: test accuracy:{:.3f}'.format(np.std(test_acc)))
        # self.slog.info('std test AUC:{:.3f}'.format(np.std(test_score)))
    
    def evaluate(self):
        test_loader = self.load_dataset(const.DataType[1],shuffle=False,index='cluster')
        test_accuracy = 0
        test_num = 0
        y_true = []
        y_prob = []
        predictions = []
        y_non1hot = []
        with torch.no_grad():
            for i,(x,y,_) in enumerate(test_loader):
                x = self.get_x_device(x)
                y = y.to(self.device)
                output = self.model_execute(x)
                prediction = torch.argmax(output,dim=1)
                predictions.extend(prediction.detach().cpu().numpy())
                y_non1hot.extend(y.detach().cpu().numpy())
                test_accuracy += (torch.sum(torch.argmax(output,dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                label = label_binarize(y.detach().cpu().numpy(),classes=np.arange(nc))
                if self.num_classes == 2:
                    label = label[:,:2]
                y_true.append(label)
        y_prob = np.concatenate(y_prob,axis = 0)
        y_true = np.concatenate(y_true,axis=0)
        # predictions = np.concatenate(predictions,axis=0)
        # y_non1hot = np.concatenate(y_non1hot,axis=0)
        if np.isnan(y_prob).any() == True:
            self.check_cluster_model() 
        auc_score = metrics.roc_auc_score(y_true, y_prob, average='micro')
        micro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        macro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        weighted_f1 = metrics.f1_score(y_non1hot, predictions, average='weighted')
        return test_accuracy,(auc_score,micro_f1,macro_f1,weighted_f1),test_num

    def get_preds(self,output):
        if isinstance(output,tuple):
            return output[0]
        return output
    
    def opt_portfolio(self,clients, Lambda = 0.5):
        W = self.reweigh(clients)
        profit = 0.0
        sigma_bar = 0
        if Global.current_round <= 1:
            return 0
        
        for i in range(len(clients)):
            Wi = W[clients[i].serial_id]
            profit += Wi * clients[i].get_score_gap

            for j in range(i + 1,len(clients)):
                Wj = W[clients[j].serial_id]
                ci,cj = min(clients[i].serial_id, clients[j].serial_id), max(clients[i].serial_id, clients[j].serial_id)

                self.TopKstore(ci,cj)
                covij = self.TopKCOV(ci,cj)
                
                sigma_bar += Wi * Wj * covij
        
        return (Lambda * sigma_bar) - ((1 - Lambda) * profit)
    
    def TopKstore(self,ci,cj):
        if not self.risk_storer[ci][cj]:
            self.risk_storer[ci][cj] = queue.Queue(maxsize = 3)
        self.risk_storer[ci][cj].put((self.sid2clients[ci].get_score_gap, self.sid2clients[cj].get_score_gap))
    
    def TopKCOV(self,ci,cj):
        q = list(self.risk_storer[ci][cj].queue)
        avg_i, avg_j = 0.0, 0.0
        for item in q:
            avg_i += item[0]
            avg_j += item[1]
        
        avg_i /= len(q)
        avg_j /= len(q)
        
        final_score = 0
        for item in q:
            final_score += ((item[0] - avg_i) * (item[1] - avg_j))
        
        final_score /= (1/max(1,len(q) - 1))
        return final_score
    
    def optimal_portfolio(self,clients,lamda = 0.5):
        total_samples = self.cluster_samples
            
        total_profit = 0.0
        sigma_bar = 0
        if Global.current_round <= 1:
            return 0
        # print(len(self.clients))
        for i in range(len(self.clients)):
            # Wi = clients[i].train_exclude_missing_samples / total_samples
            Wi = self.clients[i].train_samples / total_samples
            total_profit += Wi * self.clients[i].get_score_gap
            for j in range(i+1,len(self.clients)):
                a, b = sorted([self.clients[i].serial_id, self.clients[j].serial_id])
                # Wj = clients[j].train_exclude_missing_samples / total_samples
                Wj = self.clients[j].train_samples / total_samples
                # Compute the covariance of client i & j
                avgi,avgj = self.clients[i].historical_scores / Global.current_round,self.clients[j].historical_scores / Global.current_round
                if Global.current_round == 2:
                    self.risk_clients_storer[a][b] = (self.clients[i].get_score_gap - avgi) * (self.clients[j].get_score_gap - avgj)+\
                    (self.clients[i].last_score_gap - avgi) * (self.clients[j].last_score_gap - avgj)
                else:     
                    self.risk_clients_storer[a][b] =\
                    ((self.risk_clients_storer[a][b] * Global.current_round - 2) +\
                    (self.clients[i].get_score_gap - avgi) *\
                    (self.clients[j].get_score_gap - avgj)) / (Global.current_round - 1)
                # sigma_ij = (clients[i].get_score_gap - avg_scores_gap) *  (clients[j].get_score_gap - avg_scores_gap)
                sigma_bar +=  Wi * Wj * self.risk_clients_storer[a][b]
        return (lamda * sigma_bar) - ((1 - lamda) * total_profit)
    # def get_simga(self,clients,i,j,):
    
    def reweigh(self,clients):
        total_samples = 0
        for client in clients:
            total_samples += client.train_samples
        
        reset_weight = {
            client.serial_id:client.train_samples / total_samples for client in clients
        }
        return reset_weight
    
    def do_compute_banzhaf(self,clients,global_normscore):
        current_selected_samples = self.get_total_samples(clients)
        current_selected_samples = self.cluster_samples
        power_score = 0
        # drop_power_sets = dict()
        self.print_gap_scores(clients)
        for client in clients:
            power_score += (client.get_normalized_score * client.train_samples)
        # total_score = power_f1 / current_selected_samples
        # No one is important
        # if total_powerf1 < global_score:
        #     return
        # Select the important one
        if len(clients) == 1:
            self.clients_select_times[clients[0].serial_id] = self.clients_select_times.get(clients[0].serial_id,0) + 1
            if clients[0].get_normalized_score > global_normscore:
                self.importance_count[clients[0].serial_id] = self.importance_count.get(clients[0].serial_id,0) + 1
            return
        
        for client in clients:
            self.clients_select_times[client.serial_id] = self.clients_select_times.get(client.serial_id,0) + 1
            current_drop_power = (power_score - (client.get_normalized_score * client.train_samples)) / (current_selected_samples - client.train_samples)
            if current_drop_power < global_normscore or client.get_normalized_score < global_normscore:
                continue
            self.importance_count[client.serial_id] = self.importance_count.get(client.serial_id,0) + 1

    def print_gap_scores(self,clients):
        gap_scores = []
        for client in clients:
            gap_scores.append(client.get_score_gap)
        print('cluster {} clients gap scores:{}'.format(self.id,gap_scores))

    def update_cluster_probability(self,pre_prob):
        length = len(self.clients)
        avg_prob = 1 / length
        clients_probs = [0] * length
        for i,prob in enumerate(self.clients_prob_power):
            if prob == 0:
                clients_probs[i] = pre_prob
                continue
            clients_probs[i] = pre_prob *(1 + prob - avg_prob)
        values = np.array(clients_probs)
        probabilities = values/values.sum()
        return probabilities.tolist()
        
    def compute_prob_powers(self):
        self.client_normindex = dict()
        sum_normindex = 0
        for client in self.clients:
            self.client_normindex[client.serial_id] = get_normalized_value(
                power = self.importance_count.get(client.serial_id,0),
                times = self.clients_select_times.get(client.serial_id,0),
                floatPos = 4
            )
            sum_normindex += self.client_normindex[client.serial_id]
        self.clients_prob_power = [0] * len(self.clients)
        for i in range(len(self.clients)):
            if sum_normindex == 0:
                self.clients_prob_power[i] = 0
                continue
            self.clients_prob_power[i] = self.client_normindex[self.clients[i].serial_id] / sum_normindex

    def get_x_device(self,x_modalities):
        for i in range(len(x_modalities)):
            if isinstance(x_modalities[i],list):
                x_modalities[i][0] = x_modalities[i][0].to(self.device)
            else:
                x_modalities[i] = x_modalities[i].to(self.device)
        return tuple(x_modalities)
    
    def set_cluster_log(self,args):
        self.logkey = args['dataset']+'_'+datetime.now().strftime(const.LOG_DIR_TIME_FORMAT)
        log_dir = args.get(const.LOG_PATH_KEY,const.DEFAULT_LOG_DIR)
        log_path = os.path.join(log_dir,args['algorithm'],self.logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        index = const.CLUSTER_ + str(self.id)
        self.ctrlog = Attender(
            index = index,
            filePath = os.path.join(log_path,index + const.LOG_SUFFIX),
            # handlers = [ColoredConsoleHandler(bubble=True,format_string=const.LOG_FILE_FORMAT)],
        )