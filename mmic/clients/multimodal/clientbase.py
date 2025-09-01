import copy
import os
from const import constants as const
from const import tasks
import torch.nn as nn
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, SubsetRandomSampler
from fedlog.logbooker import Attender
from torch import Tensor
from utils.collate import collate_mm_fn_padd
from utils.multimodaldata import api as mmapi
from const.criterion import GetDatasetLoss 
from sklearn.preprocessing import label_binarize
from evaluator.api import DatasetEvaluator
from utils.data import read_missing_data
import const.config as constconfig
from sklearn import metrics
from algorithm import normalized
from models.criterion import gmc_loss
class MultmodalClientBase:
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        self.algorithm = args['algorithm']
        self.id = id
        self.serial_id = serial_id
        self.logkey = logkey
        self.set_client_log(args,logkey)
        #define models
        # self.model = copy.deepcopy(args['model'])
        self.merge_models(args)
        self.dataset = args['dataset']
        self.device = args['device']
        
        if 'save_dir' in args.keys():
            self.save_dir = os.path.join(os.getcwd(),args['save_dir'])
        if 'num_classes' in args.keys():
            self.num_classes = args['num_classes']
            
        self.train_samples = train_samples
        self.test_samples = test_samples
        # model
        if 'batch_size' not in args:
            self.batch_size = args[args['dataset']]['batch_size']
        else:
            self.batch_size = args['batch_size']
        self.num_classes = args[args['dataset']].get('num_classes',0)
        
        self.learning_rate = args['learning_rate']
        self.epochs = args['epochs']
        self.local_epochs = args['epochs']
        # self.eval_gap = args['eval_gap']
        self.global_rounds = args['global_rounds']
        
        self.learning_rate_decay = args.get('lr_decay',True)
        self.learning_rate_decay_gamma = args.get('lr_decay_gamma',0.99)

        self.dataset_dir = const.DIR_DEPOSITORY
        if 'dataset_dir' in args.keys():
            self.dataset_dir = args['dataset_dir']
        self.is_data_preload = args[self.dataset].get('is_preload',True)
        # self.dataset_dir = os.getcwd()
        self.train_time = {'rounds':0, 'total_cost':0.0}
        self.send_time = {'rounds':0, 'total_cost':0.0}
        self.task = args.get('task',tasks.DatasetTasks[self.dataset])
        self.loss = GetDatasetLoss(self.dataset,self.task,self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.learning_rate
        )
        if self.algorithm in args['fedAlgorithm'].keys():
            self.fedalgo = args['fedAlgorithm'][self.algorithm]

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=self.learning_rate_decay_gamma
        )
        evaluator = DatasetEvaluator.get(self.dataset)
        if evaluator is not None:
            self.evaluator =  evaluator(
                logger = self.clog,
                logkey = logkey,
            )
        self.train_subrate = args.get('training_subset_rate',1.0)
        self.test_subrate = args.get('test_subset_rate',1.0)
        self.advanced_score = 0
        self.last_updated_score = 0
        self.train_missing_modalities = [0] * len(constconfig.DatasetModalities[self.dataset])
        self.sub_train_samples = int(self.train_subrate * self.train_samples)
        self.finalloss = 0
        
        self.missing_file_name = self.gen_missing_filename(args)
        self.tracker = args['gpu_tracker']
        # self.old_model = None
    
    def __set_func(self):
        self.read_missing_data = read_missing_data
    
    def merge_models(self,args):
        self.need_base_model = False
        if args['model_name'] == 'clip' or args['model_name'] == 'protoclip':
            aggregate_model = args['model']
            if args['models'][args['model_name']].get('partial',True):
                self.model = copy.deepcopy(aggregate_model['partial'])
            if args['models'][args['model_name']].get('pretrained',True):
                self.need_base_model = True
                self.basemodel = aggregate_model['pretrained']
                self.model._init_architecture(self.basemodel)
        else:
            self.model = copy.deepcopy(args['model'])
    
    def gen_missing_filename(self,args):
        missing_rate_prefix = 'missing'
        if 'missing_modal_rate' in args[self.dataset]:
            missing_rate_prefix += '_' + str(args[self.dataset]['missing_modal_rate'])
        if 'missing_modal_clients_rate' in args[self.dataset]:
            missing_rate_prefix += '_' + str(args[self.dataset]['missing_modal_clients_rate'])
        return missing_rate_prefix + const.PICKLE_SUFFIX
    
    def load_dataset(self,data_type = const.DataType[0],shuffle = True, batch_size=10,index = 'cluster'):
        # if batch_size == None:
        if self.batch_size is not None:
            batch_size = self.batch_size
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        missing_maps = {}
        if data_type == const.DataType[0]:
            missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),idx=self.serial_id,missing_file=self.missing_file_name)
        collate_fn = mmapi.CollateGenerator[self.dataset](self.serial_id,missing_maps)
        sampler = None
        if (data_type == const.DataType[0] and self.train_subrate < 1.0) or (data_type == const.DataType[1] and self.test_subrate < 1.0):
            sub_rate = self.train_subrate if data_type == const.DataType[0] else self.test_subrate
            sub_samples = int(len(train_data) * sub_rate)
            random_indices = np.random.choice(len(train_data), sub_samples, replace=False)
            sampler = SubsetRandomSampler(random_indices)
            shuffle = False
        drop_last = (data_type != const.PREFIX_TEST)
        return DataLoader(train_data,batch_size=self.batch_size,drop_last=drop_last, shuffle=shuffle,collate_fn=collate_fn,sampler=sampler)

    def train(self):
        self.model.train()
        trainloader = self.load_dataset(
            data_type = const.PREFIX_TRAIN,
            shuffle = True,
            index = self.serial_id,
        )
        start_time = time.time()

        max_local_epochs = self.local_epochs
        finalloss = 0
        #Here we need to check whether is 
        self.already_check = False
        self.train_missing_modalities = [0] * len(constconfig.DatasetModalities[self.dataset])
        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for _ in range(max_local_epochs):
            if self.task == tasks.TASK_CLASSIFY: 
                finalloss = self.norm_train(trainloader)
            elif self.task == tasks.TASK_RETRIEVAL:
                finalloss = self.retrieval_train(trainloader)
        # self.model.cpu()
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        # self.model_update_check()
        self.train_time['rounds'] += 1
        self.train_time['total_cost'] += time.time() - start_time
        self.finalloss = finalloss
        self.clog.debug('round: {}, finalloss:{}'.format(self.train_time['rounds'],finalloss))
    
    def check_modalities(self,x_modalities):    
        if self.already_check:
            return
        x1,x2 = x_modalities[0],x_modalities[1]
        for i in range(self.batch_size):
            if torch.all(x1[i] == 0):
                self.train_missing_modalities[0] += 1
            if torch.all(x2[i] == 0):
                self.train_missing_modalities[1] += 1
        self.clog.info('missing modality nums {}'.format(self.train_missing_modalities))
        self.already_check = True
        
    def norm_train(self,dataloader):
        # minloss = 1 << 32
        finalloss = 0
        train_missing_modal_maps = {}
        # self.old_model = copy.deepcopy(self.model)
        missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),self.serial_id,missing_file=self.missing_file_name)
        # self.tracker.track()
        for _, (x, y, indexs) in enumerate(dataloader):
            x_modalities = self.get_x_device(x)
            y = y.to(self.device)
            # self.check_modalities(indexs)
            # if self.train_slow:
            #     time.sleep(0.1 * np.abs(np.random.rand()))
            outcome = self.model_execute(x_modalities)
            loss = self.loss(outcome, y)
            self.optimizer.zero_grad()
            loss.backward()
            finalloss = loss.item()
            self.optimizer.step()
            self.check_trainbatch_missing(indexs,missing_maps,train_missing_modal_maps)
        # self.tracker.track()
        return finalloss
    
    def retrieval_train(self,dataloader):
        # minloss = 1 << 32
        train_missing_modal_maps = {}
        # self.old_model = copy.deepcopy(self.model)
        missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),self.serial_id,missing_file=self.missing_file_name)
        for _, x in enumerate(dataloader):
            x_modalities = self.get_x_device(x)
            self.check_modalities(x_modalities)
            outcome = self.model_execute(x_modalities)
            loss = self.loss(outcome.t())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            self.check_trainbatch_missing(x_modalities[-1],missing_maps,train_missing_modal_maps)
            # minloss = min(minloss,loss.item())
        self.train_missing_modal_maps = train_missing_modal_maps
        return loss.item()
    
    def model_update_check(self):
        if self.old_model is None:
            return
        old_state_dict = self.old_model.state_dict()
        state_dict = self.model.state_dict()
        params_changes = {}
        for (name,old_param),(_,new_param) in zip(old_state_dict.items(),state_dict.items()):
            params_changes[name] = torch.sum(torch.abs(new_param - old_param))

    def check_trainbatch_missing(self,idxs,missing_maps,train_missing_modal_maps):
        if missing_maps is None:
            return
        for id in idxs:
            if id in missing_maps:
                train_missing_modal_maps[missing_maps[id]] = train_missing_modal_maps.get(missing_maps[id],0) + 1
    
    # def get_x_device(self,x_modalities):
    #     x_modalities = list(x_modalities)
    #     for i in range(len(x_modalities)):
    #         if isinstance(x_modalities[i],list):
    #             x_modalities[i][0] = x_modalities[i][0].to(self.device)
    #         else:
    #             if isinstance(x_modalities[i],tuple):
    #                 x_modalities[i] = torch.Tensor(list(x_modalities[i]))
    #             x_modalities[i] = x_modalities[i].to(self.device)
    #     return tuple(x_modalities)

    def get_x_device(self, x_modalities):
        x_modalities = list(x_modalities)
        for i in range(len(x_modalities)):
            if isinstance(x_modalities[i], list):
                x_modalities[i][0] = x_modalities[i][0].to(self.device)
            else:
                if isinstance(x_modalities[i], tuple):
                    # torch.tensor and preserve original model
                    x_modalities[i] = torch.tensor(list(x_modalities[i]), dtype=torch.float32 if isinstance(x_modalities[i][0], float) else torch.int64)
                x_modalities[i] = x_modalities[i].to(self.device)
        return x_modalities
    
    def get_preds(self,output):
        if isinstance(output,tuple):
            return output[0]
        return output
    
    def set_client_log(self, args, logkey):
        log_dir = const.DEFAULT_LOG_DIR
        if const.LOG_PATH_KEY in args.keys():   
            log_dir = args[const.LOG_PATH_KEY]
        log_path = os.path.join(log_dir,self.algorithm,logkey)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.clog = Attender(
            index = const.CLIENT_+self.id,
            filePath = os.path.join(log_path,const.CLIENT_+self.id+const.LOG_SUFFIX),
        )
    
    def test_metrics(self):
        '''
            Multimodal haven't support client test metrics 
        '''
        pass

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
    
    def validate(self,task):
        self.clog.debug('{}{}{}'.format('-'*20, self.global_rounds, '-'*20))
        if task == tasks.TASK_CLASSIFY:
            self.norm_validate()
        elif task == tasks.TASK_RETRIEVAL:
            self.retrieval_validate()
    
    def norm_validate(self,testloader = None):
        test_metrics_res = self.client_evaluate(testloader)
        test_acc = test_metrics_res[0] * 1.0 / test_metrics_res[2]
        test_score = test_metrics_res[1]
        self.advanced_score, self.last_updated_score = test_score[1] * 100, self.advanced_score
        
        # self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        self.clog.info('server: accuracy:{:.3f}'.format(test_acc*100))
        self.clog.info('server: avg test auc:{:.3f}, micro_f1{} ,macro_f1:{}, weighted_f1:{}'.format(test_score[0],test_score[1],test_score[2],test_score[3]))
        
        self.clog.info('std: test accuracy:{:.3f}'.format(np.std(test_acc)))
        self.clog.info('std test AUC:{:.3f}'.format(np.std(test_score)))
    
    def retrieval_validate(self,test_loader = None):
        if test_loader is None:
            test_loader = self.load_dataset(const.DataType[1], shuffle=False, index=constconfig.GetDatasetTestIndex(dataset=self.dataset,index=self.serial_id))
        self.model.eval()
        res = self.evaluator.evalrank(self.model,test_loader)
        # self.advanced_score = res['rsum']
        self.advanced_score, self.last_updated_score = res['rsum'], self.advanced_score
    
    def client_evaluate(self,test_loader = None):
        if test_loader is None:
            test_loader = self.load_dataset(const.DataType[1],shuffle=False,index=constconfig.GetDatasetTestIndex(dataset=self.dataset,index=self.serial_id))
        self.model.eval()
        test_accuracy = 0
        test_num = 0
        y_prob = []
        y_true = []
        predictions = []
        y_non1hot = []
        with torch.no_grad():
            for i,(x,y,_) in enumerate(test_loader):
                x = self.get_x_device(x)
                y = y.to(self.device)
                outcome = self.model_execute(x)
                prediction = torch.argmax(outcome,dim=1)
                predictions.extend(prediction.detach().cpu().numpy())
                y_non1hot.extend(y.detach().cpu().numpy())
                test_accuracy += (torch.sum(torch.argmax(outcome,dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(outcome.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                label = label_binarize(y.detach().cpu().numpy(),classes=np.arange(nc))
                if self.num_classes == 2:
                    label = label[:,:2]
                y_true.append(label)
        y_prob = np.concatenate(y_prob,axis = 0)
        y_true = np.concatenate(y_true,axis = 0)
        # predictions = np.concatenate(predictions,axis=0)
        # y_non1hot = np.concatenate(y_non1hot,axis=0)
        auc_score = metrics.roc_auc_score(y_true,y_prob,average='micro')
        micro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        macro_f1 = metrics.f1_score(y_non1hot,predictions, average='macro')
        weighted_f1 = metrics.f1_score(y_non1hot, predictions, average='weighted')
        return test_accuracy,(auc_score,micro_f1,macro_f1,weighted_f1),test_num

    def model_execute(self,X,model = None,only_preds = True,**kwargs):
        model = model if model is not None else self.model 
        if self.need_base_model:
            outcome = model(X,self.basemodel,**kwargs)
        else:
            outcome = model(X,**kwargs)
        if only_preds:
            return self.get_preds(outcome)
        else:
            return outcome
        
    def __hash__(self):
        return hash(self.serial_id)
    