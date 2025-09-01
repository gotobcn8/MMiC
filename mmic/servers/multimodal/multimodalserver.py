import const.constants as const
from datetime import datetime
import os
from fedlog.logbooker import Attender
from fedlog.logbooker import ColoredConsoleHandler
from utils.multimodaldata.dataloader import read_multimodal_data
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from utils.collate import collate_mm_fn_padd
import time
import torch 
import copy
import numpy as np
import h5py
from utils.data import read_client_data
from container import food101loader
from container import containerapi
from const import tasks
from container import retrieval
from utils.multimodaldata import api as mmapi
from evaluator import api as evalapi 
import pickle
from transformers import CLIPProcessor
from const.settings import Global
import os
import torch
from torch.utils.tensorboard import SummaryWriter

class Server:
    def __init__(self,args) -> None:
        self.args = args
        self.algorithm = args['algorithm']
        if self.algorithm in args['fedAlgorithm'].keys():
            self.fedAlgorithm = args['fedAlgorithm'][self.algorithm]
        # define log
        # 获取当前时间  
        self.writer = SummaryWriter(log_dir = os.path.join(os.getenv("HOME"),'data','fed'))
        self.set_server_log(args)
        self.args = args
        self.merge_models()
        self.device = args['device']
        self.dataset = args['dataset']
        self.num_clients = args['num_clients']
        self.join_ratio = args['join_ratio']
        self.client_drop_rate = args['client_drop_rate']
        self.learning_rate = args['learning_rate']
        
        self.global_rounds = args['global_rounds']
        Global.total_rounds = args['global_rounds']
        self.time_threthold = args['time_threthold']
        self.global_test_subrate = args.get('global_test_subrate',1.0)
        
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.server_test_acc = []
        self.server_test_auc = []
        
        self.dataset_dir = const.DIR_DEPOSITORY
        if 'dataset_dir' in args.keys():
            self.dataset_dir = args['dataset_dir']
        self.is_data_preload = args[self.dataset].get('is_preload',True)
        # clients settings
        self.late_clients = []
        self.all_clients = []
        self.clients = []
        self.new_clients_settings = args['new_clients']
        self.random_clients_selected = args['random_clients_selected']
        self.new_clients_rate = self.new_clients_settings['rate']
        self.set_for_new_clients()
        self.num_original_clients = self.num_clients - self.num_new_clients
        self.num_join_clients = self.num_original_clients * self.join_ratio
        self.num_new_clients = int(self.num_clients * self.new_clients_rate)
        
        
        self.algorithm = args['algorithm']
        self.eval_gap = args['eval_gap']
        self.budget = []
        
        self.save_dir = args.get('save_dir','results')
        if 'save_dir' in args.keys() and args['save_dir'] != '':
            self.save_models_dir = args['save_dir']
            self.save_dir = args['save_dir']
        
        # model
        if 'batch_size' not in args:
            self.batch_size = args[args['dataset']]['batch_size']
        else:
            self.batch_size = args['batch_size']
        
        self.num_classes = args[args['dataset']].get('num_classes',0)

        self.old_global_model = None
        
        self.task = args.get('task',tasks.DatasetTasks[self.dataset])
        evaluator = evalapi.DatasetEvaluator.get(self.dataset)
        if evaluator is not None:
            self.evaluator =  evaluator(
                logger = self.slog,
                logkey = self.logkey,
                save_path=self.save_dir
            )
        self.each_clients_select_records = [0] * self.num_clients

        self.__init_visualized_collector()
        self.tracker = args['gpu_tracker']        
        
    def merge_models(self):
        self.need_base_model = False
        if self.args['model_name'] == 'clip' or self.args['model_name'] == 'protoclip':
            aggregate_model = self.args['model']
            if self.args['models'][self.args['model_name']].get('partial',True):
                self.global_model = copy.deepcopy(aggregate_model['partial'])
            if self.args['models'][self.args['model_name']].get('pretrained',True):
                self.need_base_model = True
                self.basemodel = aggregate_model['pretrained']
                self.global_model._init_architecture(self.basemodel)
        else:
            self.global_model = copy.deepcopy(self.args['model'])
    
    def set_for_new_clients(self):
        # whether the new clients join setting is openning, otherwise no clients join in future.
        if not self.new_clients_settings['enabled']:
            self.new_clients_rate = 0
        # total new clients
        self.num_new_clients = int(self.num_clients * self.new_clients_rate)
        # which round that have new clients to join in
        self.start_new_joining_round = self.new_clients_settings['started_round']
        # how many clients join in each round
        self.num_new_join_each_round = self.new_clients_settings['num_join_each_round']
    
    def set_server_log(self,args):
        self.logkey = args['dataset']+'_'+datetime.now().strftime(const.LOG_DIR_TIME_FORMAT)
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():   
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir,self.algorithm,self.logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.slog = Attender(
            index = const.SERVER_KEY,
            filePath = os.path.join(log_path,const.SERVER_KEY+const.LOG_SUFFIX),
            handlers = [ColoredConsoleHandler(bubble=True,format_string=const.LOG_FILE_FORMAT)],
        )

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
            
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            # self.check_global_model()
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))
        
        # self.save_running_process()
        self.save_results()
        self.save_global_model()
    
    def check_global_model(self):
        if self.old_global_model is None:
            return
        old_state_dict = self.old_global_model.state_dict()
        state_dict = self.global_model.state_dict()
        params_changes = {}
        for (name,old_param),(_,new_param) in zip(old_state_dict.items(),state_dict.items()):
            params_changes[name] = torch.sum(torch.abs(new_param - old_param))
        print(params_changes)
        
    def save_global_model(self):
        model_path = os.path.join(self.save_dir,self.dataset,self.logkey)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path,self.algorithm + '_server' + '.pt')
        print('server saving directory:',model_path)
        torch.save(self.global_model,model_path)
    
    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time['rounds'] += 1
            client.send_time['total_cost'] += 2 * (time.time() - start_time)
    
    def set_clients(self,clientObj):
        if not self.is_data_preload:
            self.args['processer'] = CLIPProcessor
        for i in range(self.num_original_clients):
            train_data = mmapi.DatasetGenerator[self.dataset](dataset_name = self.dataset,client_sid = i,dataset_dir = self.dataset_dir,data_type = const.DataType[0],is_preload = self.is_data_preload)
            # train_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=True)
            # test_data = read_multimodal_data(self.dataset,i,self.dataset_dir,is_train=False)
            client = clientObj(
                self.args,
                id = const.ORIGINAL+str(i),
                serial_id = i,
                train_samples = len(train_data),
                test_samples = 0,
                logkey = self.logkey,
            )
            self.clients.append(client)
        self.all_clients.extend(self.clients)

    def aggregate_parameters(self):
        self.old_global_model = copy.deepcopy(self.global_model)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w,client_model in zip(self.uploaded_weights,self.uploaded_models):
            self.add_parameters(w,client_model)

    def add_parameters(self,w,client_model):
        for server_param,client_param in zip(self.global_model.parameters(),client_model.parameters()):
            server_param.data += client_param.data.clone() * w
            
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
    
    def __init_visualized_collector(self):
        self.important_layers_by_rounds = []
        self.missing_modalities_by_rounds = []
        self.global_scores = []
        self.per_avg_scores = []
        self.global_accuracy = []
        
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
                hf.create_dataset('server_test_acc', data=self.server_test_acc)
                hf.create_dataset('server_test_auc', data=self.server_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        self.slog.info('rs_test_acc', self.rs_test_acc,'rs_test_auc', self.rs_test_auc,'rs_train_loss',self.rs_train_loss)
    
    def load_dataset(self,data_type = const.PREFIX_TRAIN,shuffle = True, batch_size = 10,index = 'server'):
        # if batch_size == None:
        if self.batch_size is not None:
            batch_size = self.batch_size
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        # if not self.is_data_preload:
        #     train_data = containerapi.Containers[self.dataset](train_data[0],train_data[1])
        collate_fn = mmapi.CollateGenerator[self.dataset]()
        drop_last = (data_type != const.PREFIX_TEST)
        random_indices = np.random.choice(len(train_data), int(self.global_test_subrate * len(train_data)), replace=False)
        sampler = SubsetRandomSampler(random_indices)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=drop_last, shuffle=shuffle,collate_fn=collate_fn,sampler = sampler)
    
    def server_evaluate(self):
        # self.tracker.track()
        val_loader = self.load_dataset(const.DataType[1],shuffle=False,index='server')
        self.global_model.eval()
        test_accuracy = 0
        test_num = 0
        y_prob = []
        y_true = []
        predictions = []
        y_non1hot = []
        with torch.no_grad():
            for i,(x,y,_) in enumerate(val_loader):
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
                    nc += 10
                label = label_binarize(y.detach().cpu().numpy(),classes=np.arange(nc))
                if self.num_classes == 2:
                    label = label[:,:2]
                y_true.append(label)
        y_prob = np.concatenate(y_prob,axis = 0)
        y_true = np.concatenate(y_true,axis=0)
        # predictions = np.concatenate(predictions,axis=0)
        # y_non1hot = np.concatenate(y_non1hot,axis=0)
        # if y_prob.any() == np.nan:
        #     self.check_global_model()
        auc_score = metrics.roc_auc_score(y_true,y_prob,average='micro')
        micro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        macro_f1 = metrics.f1_score(y_non1hot,predictions, average='macro')
        weighted_f1 = metrics.f1_score(y_non1hot, predictions, average='weighted')
        # self.tracker.track()
        return test_accuracy,(auc_score,micro_f1,macro_f1,weighted_f1),test_num

    def get_preds(self,output):
        if isinstance(output,tuple):
            return output[0]
        return output
            
    def validate_interface(self):
        if self.task == tasks.TASK_CLASSIFY:
            self.norm_validate()
        elif self.task == tasks.TASK_RETRIEVAL:
            self.retrieval_validate()

    def retrieval_validate(self):
        self.global_model.eval()
        server_test_data = self.load_dataset(
            data_type = const.PREFIX_TEST,
            shuffle = False,
            index = 'server',
        )
        with torch.no_grad():
            res = self.evaluator.evalrank(self.global_model,server_test_data)
        self.global_scores.append(res['rsum'])
        
    def norm_validate(self):
        test_metrics_res = self.server_evaluate()
        test_acc = test_metrics_res[0] * 1.0 / test_metrics_res[2]
        test_auc = test_metrics_res[1]
        
        self.server_test_acc.append(test_acc)
        self.server_test_auc.append(test_auc)
        #This should be the criterion
        self.global_scores.append(test_auc[1]*100)
        self.global_accuracy.append(test_acc*100)
        # self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        self.slog.debug(f'Completed round {len(self.budget)} validation')
        self.slog.info('server: accuracy:{:.3f}'.format(test_acc*100))
        self.slog.info('server: avg test auc:{:.3f}, micro_f1:{:.3f} ,macro_f1:{:.3f}, weighted_f1:{:.3f}'.format(test_auc[0],test_auc[1]*100,test_auc[2]*100,test_auc[3]*100))
        
        self.writer.add_scalar('AUC',test_auc[0],Global.current_round)
        self.writer.add_scalar('Test Acc',test_acc,Global.current_round)
        # self.slog.info('std: test accuracy:{:.3f}'.format(np.std(test_acc)))
        # self.slog.info('std test AUC:{:.3f}'.format(np.std(test_auc)))
    
    def save_collector(self):
        save_path = os.path.join(self.save_dir,self.dataset+'_'+self.algorithm,self.logkey)
        self.save_dir = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        important_layer_collector = {
            'important_layer':self.important_layers_by_rounds,
            'missing_modalities':self.missing_modalities_by_rounds,
            'global_scores': self.global_scores,
            'per_avg_scores':self.per_avg_scores,
            'global_accuracy':self.global_accuracy,
        }
        pkl_file = os.path.join(save_path,'collector.pkl')
        self.slog.info(f'collector file:{pkl_file}')
        with open(pkl_file,mode='wb') as f:
            pickle.dump(important_layer_collector,file = f)
    
    def attend_clients_validate(self):
        avg_clients_score = 0
        total_samples = 0
        for attend_client in self.selected_clients:
            attend_client.validate(self.task)
            avg_clients_score += attend_client.advanced_score * attend_client.train_samples
            total_samples += attend_client.train_samples
        avg_clients_score /= total_samples
        self.slog.info('avg clients score: {:.3f}'.format(avg_clients_score))  
        self.per_avg_scores.append(avg_clients_score)    
        self.writer.add_scalar('Personalized',avg_clients_score,Global.current_round)      
        # current_cluster_loss = {}
    
    def get_x_device(self,x_modalities):
        for i in range(len(x_modalities)):
            if isinstance(x_modalities[i],list):
                x_modalities[i][0] = x_modalities[i][0].to(self.device)
            else:
                x_modalities[i] = x_modalities[i].to(self.device)
        return tuple(x_modalities)
    
    
    def select_clients(self,is_late_attended = False):
        self.slog.info('Starting select clients for server')
        if self.random_clients_selected:
            #random number of attend clients
            self.current_num_join_clients = np.random.choice(int(self.num_original_clients * self.join_ratio),self.num_original_clients+1)
        else:
            #static number of attend clients
            self.current_num_join_clients = len(self.clients) * (1 - self.client_drop_rate) * self.join_ratio
        selected_clients = list(np.random.choice(self.clients,int(self.current_num_join_clients),replace=False))
        for attend_client in selected_clients:
            self.each_clients_select_records[attend_client.serial_id] += 1
        return selected_clients
    
    def model_execute(self,X,model = None,only_preds = True):
        model = model if model is not None else self.global_model 
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