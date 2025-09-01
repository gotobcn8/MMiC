import copy
from fedlog.logbooker import glogger
import const.constants as const
from evaluator.api import DatasetEvaluator
import const.constants as const
from fedlog.logbooker import Attender
from utils.multimodaldata import api as mmapi
from datetime import datetime
import os
import torch
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from algorithm.normalized import get_normalized_value
import const.tasks as tasks
from torch.utils.data import DataLoader,SubsetRandomSampler
import numpy as np
class ClusterBase():
    def __init__(self,args,id,cluster_ids,clients) -> None:
        self.id = id
        self.clients = clients
        self.cluster_ids = cluster_ids
        # self.cluster_model = copy.deepcopy(args['model'])
        self.cluster_test_subrate = args['cluster'].get('cluster_test_subrate',1.0)
        self.merge_models(args)
        self.old_cluster_model = None
        self.get_cluster_sampels()
        
    def get_cluster_sampels(self):
        self.cluster_samples = 0
        for client in self.clients:
            self.cluster_samples += client.train_samples
    
    def merge_models(self,args):
        self.need_base_model = False
        if args['model_name'] == 'clip':
            aggregate_model = args['model']
            if args['models']['clip'].get('partial',True):
                self.cluster_model = copy.deepcopy(aggregate_model['partial'])
            if args['models']['clip'].get('pretrained',True):
                self.need_base_model = True
                self.basemodel = aggregate_model['pretrained']
                self.cluster_model._init_architecture(self.basemodel)
        else:
            self.cluster_model = copy.deepcopy(args['model'])
    
    def aggregate_parameters(self):
        self.old_cluster_model = copy.deepcopy(self.cluster_model)
        self.cluster_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.cluster_model.parameters():
            param.data.zero_()
        
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)
    
    def add_parameters(self,w,client_model):
        for cluster_param,client_param in zip(self.cluster_model.parameters(),client_model.parameters()):
            cluster_param.data += client_param.data.clone() * w
    
    def set_parameters(self,global_model):
        for new_param, old_param in zip(global_model.parameters(), self.cluster_model.parameters()):
            old_param.data = new_param.data.clone()
    
    def check_cluster_model(self):
        if self.old_cluster_model is None:
            return
        old_state_dict = self.cluster_model.state_dict()
        state_dict = self.cluster_model.state_dict()
        params_changes = {}
        for (name,old_param),(_,new_param) in zip(old_state_dict.items(),state_dict.items()):
            params_changes[name] = torch.sum(torch.abs(new_param - old_param))
        print(params_changes)
    
    def receive_models(self,selected_clients):
        if len(selected_clients) <= 0:
            glogger.warn("the cluster clients selected is 0")
            return
        
        self.uploaded_cids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_samples = 0
        for client in selected_clients:
            try:
                avg_train_time_cost = client.train_time['total_cost'] / client.train_time['rounds']
                avg_send_time_cost = client.send_time['total_cost'] / client.send_time['rounds']
                client_time_cost = avg_train_time_cost + avg_send_time_cost
            except ZeroDivisionError:
                client_time_cost = 0
            # if client_time_cost > self.time_threthold:
            #     continue
            total_samples += client.train_samples
            self.uploaded_cids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        
        for i,w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / total_samples
    
    def test_model_generalized(self,test_client):
        '''
        This function is used to test the personalized model generalized ability in cluster. 
        '''
        glogger.debug('Starting to evaluate the generalization of cluster {}'.format(self.id))
        avg_test_acc,total_test_num,avg_auc = 0,0,0
        if len(self.clients) <= 1:
            glogger.warn('test_client:{}, cluster clients less than 1,can not test the generallization'.format(test_client.id))
            return
        if test_client.train_time['rounds'] == 0:
            glogger.info('this is first time to test the generalization of {}'.format(test_client.id))

        for client in self.clients:
            if test_client.id == client.id:
                continue
            test_acc,test_num,auc = client.test_other_model(test_client.model)
            glogger.info('cluster {} new coming client {} be tested client {}'.format(self.id,test_client.id,client.id))
            glogger.info('test_accuracy:{:.3f}% test_num:{} correct_num:{} test_auc:{}'.format((test_acc*100.0)/test_num,test_num,test_acc,auc))
            total_test_num += test_num
            avg_test_acc += test_acc
            avg_auc += auc*test_num
        
        # if avg_test_acc == 0:
        avg_test_acc =  (avg_test_acc * 1.0) / total_test_num 
        avg_auc =  (avg_auc * 1.0) / total_test_num
        glogger.info('-------avg_acc:{:.3f}% avg_auc:{:.3f}'.format(avg_test_acc*100,avg_auc))
    
    def get_total_samples(self,clients,exclude_missing = True):
        total_samples = 0
        for client in clients:
            if exclude_missing:
                total_samples += client.train_exclude_missing_samples
            else:
                total_samples += client.sub_train_samples
        return total_samples
    
    def get_training_loss(self,clients):
        total_train_loss = 0
        total_sampels = self.get_total_samples(clients,exclude_missing = False)
        for client in clients:
            total_train_loss += (client.sub_train_samples / total_sampels) * client.finalloss
        return total_train_loss

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
    
    def model_execute(self,X,model = None,only_preds = True):
        model = model if model is not None else self.cluster_model 
        if self.need_base_model:
            outcome = model(X,self.basemodel)
        else:
            outcome = model(X)
        if only_preds:
            return self.get_preds(outcome)
        else:
            return outcome
    
    def get_preds(self,output):
        if isinstance(output,tuple):
            return output[0]
        return output

class EvalCluster(ClusterBase):
    def __init__(self,args,id,model,clients = [],cluster_ids = []) -> None:
        super().__init__(args,id=id,cluster_ids=cluster_ids,clients=clients)
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
        self.num_classes = args[self.dataset].get('num_classes',0)
        self.is_data_preload = args[self.dataset].get('is_preload',True)
        self.device = args['device']
        self.set_cluster_log(args)
        evaluator = DatasetEvaluator.get(self.dataset)
        if evaluator is not None:
            self.evaluator =  evaluator(
                logger = self.ctrlog,
                logkey = self.logkey,
            )
    
    def validate(self,task):
        if task == tasks.TASK_CLASSIFY:
            self.norm_validate()
        elif task == tasks.TASK_RETRIEVAL:
            self.retrieval_validate()
            
    def retrieval_validate(self):
        test_loader = self.load_dataset(const.DataType[1],shuffle=False,index='cluster')
        self.cluster_model.eval()
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
    def get_x_device(self,x_modalities):
        for i in range(len(x_modalities)):
            if isinstance(x_modalities[i],list):
                x_modalities[i][0] = x_modalities[i][0].to(self.device)
            else:
                x_modalities[i] = x_modalities[i].to(self.device)
        return tuple(x_modalities)
    
    def evaluate(self):
        test_loader = self.load_dataset(const.DataType[1],shuffle=False,index='cluster')
        self.cluster_model.eval()
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
                prediction = torch.argmax( output,dim=1)
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
        auc_score = metrics.roc_auc_score(y_true, y_prob, average='micro')
        micro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        macro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        weighted_f1 = metrics.f1_score(y_non1hot, predictions, average='weighted')
        return test_accuracy,(auc_score,micro_f1,macro_f1,weighted_f1),test_num

    def load_dataset(self,data_type = const.PREFIX_TRAIN,shuffle = True, batch_size=10,index = 'cluster'):
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        collate_fn = mmapi.CollateGenerator[self.dataset]()
        drop_last = (data_type != const.PREFIX_TEST)
        random_indices = np.random.choice(len(train_data), int(self.cluster_test_subrate * len(train_data)), replace=False)
        sampler = SubsetRandomSampler(random_indices)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=drop_last, shuffle=shuffle,collate_fn=collate_fn,sampler=sampler)

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
