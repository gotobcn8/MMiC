from .clientbase import MultmodalClientBase
from utils.multimodaldata.dataloader import load_multimodal_data
from algorithm.sim.lsh import ReflectSketch
import utils.multimodaldata.dataloader as mmloader
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from utils.fedtensor import reshape_with
import const.constants as const
import const.criterion as criterionconst
from sklearn.preprocessing import label_binarize
from utils.collate import collate_mm_fn_padd
import copy
import time
import torch
import gc
import numpy as np
import os
from sklearn import metrics
import const.tasks as tasks
from evaluator.api import DatasetEvaluator
from utils.data import read_missing_data
from utils.multimodaldata import api as loaderapi
from utils.multimodaldata import api as mmapi
import const.config as constconfig
from algorithm import normalized
from container.processer import pretrain
from utils import vocab
from models.criterion import gmc_loss

class MMOFCHP(MultmodalClientBase):
    def __init__(self, args, id, train_samples, test_samples, serial_id, logkey, **kwargs):
        super().__init__(args, id, train_samples, test_samples, serial_id, logkey, **kwargs)
        ofchp = args['fedAlgorithm'][self.algorithm]
        # self.model = copy.deepcopy(args['model'])
        # A metric to measure current status for the clients.
        
        self.advanced_score = 0
        self.max_score_gap = 0
        self.min_score_gap = 1 << 10 #first we set a big value
        self.shapweight = 1.0
        self.important_layer_key = ''
        # This is for the pacfl
        self.budget = ofchp.get('budget',20)
        self.partition = 'dirichlet'
        self.store_params_maps = []
        self.args = args
        self.is_gen_modality = args.get('is_gen_modality')
        self.extra_loss = gmc_loss.GMCLoss(self.device)
        self.temperature = 0.5
        self.poverty_check = args.get('poverty_check',True)
        self.historical_scores = 0
        
    def load_dataset(self,data_type = const.DataType[0],shuffle = True, batch_size=10,index = 'test'):
        # if batch_size == None:
        if self.batch_size is not None:
            batch_size = self.batch_size
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        missing_maps = {}
        if data_type == const.DataType[0]:
            missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),idx=self.serial_id,missing_file=self.missing_file_name)
        if self.is_gen_modality:
            collate_fn = mmapi.CollateGenerator[self.dataset](self.serial_id,missing_maps,t2imodel = self.args['t2i_model'],i2tmodel=self.args['i2t_model'],device=self.device)
        else:
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
        
    def count_sketch(self,hashF):
        self.reflector = ReflectSketch(
            hashF = hashF,
            dtype = float,
            data_vol = hashF.data_rows,
            hash_num = hashF.hash_num,
            dimension = hashF.dimension,
        )
        start_time = time.time()
        sketch_data = self.load_sketch_data(hashF.data_rows)
        for x in sketch_data:
            # x = x.reshape(-1,1)
            self.reflector.get_sketch(x,self.device)
        self.sketch = self.reflector.sketch
        self.minisketch = self.reflector.sketch / self.reflector.NumberData
        # count_sketch_time = time.time() - start_time
        self.clog.info('{} :calculate sketch time {:.3f}s'.format(self.id,time.time()-start_time))
        del sketch_data
        return self.minisketch
        
    def load_sketch_data(self,data_volume = 1000):
        sketch_data = loaderapi.DatasetGenerator[self.dataset](
            self.dataset,
            self.serial_id,
            self.dataset_dir,
            const.DataType[0],
            is_preload = self.is_data_preload,
        )
        data_ranges = [i for i in range(int(len(sketch_data)))]
        selected_index = np.random.choice(data_ranges,data_volume)
        dispersed_sets = self.fusion_selected_data(sketch_data, selected_index, dimension = self.reflector.dimension)
        batch_size = len(dispersed_sets)
        return DataLoader(
            dataset = dispersed_sets,
            batch_size = batch_size,
            shuffle = True
        )

    def process_raw_data(self,x):
        if self.is_data_preload:
            return
        inputs = pretrain.ClipProcessor(x[1],x[0],return_tensors='pt')
        return inputs['pixel_values']
    
    def fusion_selected_data(self,sketch_data,selected_index,dimension = 1280)->torch.Tensor:
        dispersed_sets = []
        for index in selected_index:
            if self.task == tasks.TASK_CLASSIFY: 
                x = sketch_data[index][0]
            elif self.task == tasks.TASK_RETRIEVAL:
                x = sketch_data[index] 
            if not self.is_data_preload:
                x = self.process_raw_data(x)
            # ximage = reshape_with(x[0],dimension)
            # xtext = reshape_with(x[1],dimension)
            x_concat = self.fusion_multiple_modalities(x,dimension=dimension,modal_nums=1)
            dispersed_sets.append(torch.cat(x_concat,dim = 0))
        return torch.cat(dispersed_sets)
    
    def fusion_multiple_modalities(self,x,dimension,modal_nums = 2):
        x_concat = []
        for i in range(modal_nums):
            xmodality = reshape_with(x[i],dimension)
            x_concat.append(xmodality)
        return x_concat
    
    def validate(self,task):
        self.clog.debug('{}{}{}'.format('-'*20, self.global_rounds, '-'*20))
        self.last_score_gap = self.get_score_gap
        if task == tasks.TASK_CLASSIFY:
            self.norm_validate()
        elif task == tasks.TASK_RETRIEVAL:
            self.retrieval_validate()
        self.max_score_gap = max(self.advanced_score, self.max_score_gap)
        self.min_score_gap = min(self.advanced_score, self.min_score_gap)
        self.train_exclude_missing_samples = self.train_samples
        self.accumulate_score()
        #do shapley value count
        # self.eval_shapley_scores()
    
    def retrieval_validate(self,test_loader = None):
        if test_loader is None:
            test_loader = self.load_dataset(const.DataType[1], shuffle=False, index=constconfig.GetDatasetTestIndex(dataset=self.dataset,index=self.serial_id))
        self.model.eval()
        res = self.evaluator.evalrank(self.model,test_loader)
        # self.advanced_score = res['rsum']
        self.advanced_score, self.last_updated_score = res['rsum'], self.advanced_score
        
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
    
    def client_evaluate(self,test_loader = None):
        # self.tracker.track()
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
        y_true = np.concatenate(y_true,axis = 0)
        # predictions = np.concatenate(predictions,axis=0)
        # y_non1hot = np.concatenate(y_non1hot,axis=0)
        auc_score = metrics.roc_auc_score(y_true,y_prob,average='micro')
        micro_f1 = metrics.f1_score(y_non1hot, predictions, average='macro')
        macro_f1 = metrics.f1_score(y_non1hot,predictions, average='macro')
        weighted_f1 = metrics.f1_score(y_non1hot, predictions, average='weighted')
        # self.tracker.track()
        return test_accuracy,(auc_score,micro_f1,macro_f1,weighted_f1),test_num

    def accumulate_score(self):
        self.historical_scores += self.get_score_gap
        
    
    @property
    def get_score_gap(self):
        return self.advanced_score - self.last_updated_score
    
    @property
    def get_normalized_score(self):
        return normalized.maxmin_norm(self.advanced_score,self.max_score_gap,self.min_score_gap)
    
    def eval_shapley_scores(self,shapleyloader = None,subrate = 0.2):
        self.model.eval()
        if shapleyloader is None:
            pure_data = mmapi.DatasetGenerator[self.dataset](self.dataset,const.CLUSTER_KEY,self.dataset_dir,data_type=const.DataType[1],is_preload = self.is_data_preload)
            random_indices = np.random.choice(len(pure_data), int(len(pure_data) * subrate), replace=False)
            sampler = SubsetRandomSampler(random_indices)
            shapleyloader = DataLoader(pure_data,batch_size=self.batch_size,sampler=sampler)
        #  Not shield, shield 0, shield 1 shield [0,1]
        permutations = [[],[0],[1],[0,1]]
        results = [0] * 4
        if len(constconfig.DatasetModalities[self.dataset]) == 2:
            for i,v in enumerate(permutations):
                collate_fn = mmapi.CollateGenerator[self.dataset](index = self.serial_id,missing_clients_map = None,missing_modality = v)
                shapleyloader.collate_fn = collate_fn
                results[i] = self.getEvalResults(testloader=shapleyloader)
        shap0 = results[0] - results[1] + results[2] - results[3]
        shap1 = results[0] - results[2] + results[1] - results[3]
        # self.modality_importance = [shap0]
        bottom = normalized.softplus(shap0) + normalized.softplus(shap1)
        shape0weight,shape1weight = normalized.softplus(shap0)/bottom,normalized.softplus(shap1)/bottom
        self.missing_samples = (self.train_missing_modalities[0] * shape0weight + self.train_missing_modalities[1] * shape1weight)
        self.train_exclude_missing_samples = self.sub_train_samples * 2 - self.missing_samples
        # self.shapweight = (shap0 * (self.train_samples - self.train_missing_modalities[0]) + shap1 * (self.train_samples - self.train_missing_modalities[1])) / ((self.train_subrate * self.train_samples) * 2)
        # return [shap0,shap1]
    
    def getEvalResults(self,testloader):
        if self.task == tasks.TASK_CLASSIFY:
            return self.client_evaluate(testloader)[1][1]
        elif self.task == tasks.TASK_RETRIEVAL:
            return self.evaluator.evalrank(self.model,testloader)['rsum']
    
    def train(self):
        self.model.train()
        # self.tracker.track()
        trainloader = self.load_dataset(
            data_type = const.PREFIX_TRAIN,
            shuffle = True,
            index = self.serial_id,
        )
        # self.tracker.track()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        finalloss = 0
        #Here we need to check whether is 
        self.already_check = False
        self.train_missing_modalities = [0] * len(constconfig.DatasetModalities[self.dataset])
        # self.train_missing_modal_maps = [0] * len(constconfig.DatasetModalities[self.dataset])
        # IT'S FOR CHECK MISSING MODALITIES IMPORTANCE
        self.lastmissing = False
        self.currentmissing = False
        self.checkTimes = 0
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
        self.explore_params_gap()
        self.store_params_maps.clear()
        self.train_time['rounds'] += 1
        self.train_time['total_cost'] += time.time() - start_time
        self.finalloss = finalloss
        self.clog.debug('round: {}, finalloss:{}'.format(self.train_time['rounds'],finalloss))
    
    def check_modalities(self,indexs,missing_maps,x_modalities = None):   
        missing_indexs = [3] * len(indexs) 
        if missing_maps is None:
            return missing_indexs
        for (i,index) in enumerate(indexs):
            if index in missing_maps:
                self.train_missing_modalities[missing_maps[index]] += 1
                self.currentmissing = True
                missing_indexs[i] ^= (1 << missing_maps[index])
                if self.is_gen_modality:
                    self.gen_missing_modality(x_modalities,i,missing_maps[index])
        return missing_indexs
    
    @torch.no_grad()
    def gen_missing_modality(self,x_modalities,index,modal_missing):
        '''
        x_modalities is input values.\\
        index is the index in this batch data.\\
        modal_missing is which modal batch missing.
        '''
        if x_modalities is None or modal_missing is None:
            self.clog.warn('generate missing modalities failed,because input or modal missing is None')
            return
        images,texts,attention_masks = x_modalities[0],x_modalities[1],x_modalities[2]
        if modal_missing == 0:
            t2imodel = self.args['t2i_model']
            t2imodel.eval()
            text = texts[index].clone().detach().to(self.device)
            text = text.unsqueeze(0)
            lengths = vocab.get_lengths(text)
            lengths = torch.as_tensor(lengths)
            lengths = lengths.to(self.device)
            images[index].copy_(t2imodel(text,lengths).squeeze(0).cpu())
        if modal_missing == 1:
            img = images[index].clone().detach().to(self.device)
            img = img.unsqueeze(0)
            i2tmodel = self.args['i2t_model']
            i2tmodel.eval()
            gentext = i2tmodel(img,generate_lengths = texts.size(1)).squeeze(0)
            texts[index].copy_(gentext)
            if texts[index][0] != 49406:
                texts[index][0] = 49406
            for j in range(texts.size(1)):
                if attention_masks[index][j] == 0:
                    texts[index,j-1:] = 49407
                    break
            texts[index][-1] = 49407
        
    def norm_train(self,dataloader):
        # self.old_model = copy.deepcopy(self.model)
        # self.tracker.track()
        missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),self.serial_id,missing_file=self.missing_file_name)
        for i, (x, y,indexs) in enumerate(dataloader):
            x_modalities = self.get_x_device(x)
            # self.tracker.track()
            y = y.to(self.device)
            # self.tracker.track()
            missing_indexs = self.check_modalities(indexs,missing_maps,x_modalities)
            self.copyoldmodel(i)
            # self.tracker.track()
            outcome = self.model_execute(x_modalities,only_preds = False,missing_indexs = missing_indexs)
            # self.tracker.track()
            # loss = self.loss(outcome, y)
            loss = self.extra_loss(outcome[0],y,outcome[1],self.temperature,self.batch_size,missing_indexs,False)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.tracker.track()
            self.checkMMParamsChange(i)
            self.lastmissing, self.currentmissing = self.currentmissing, False
        self.clog.debug('client:{},loss:{}'.format(self.serial_id,loss.item()))
        self.tracker.track()
        return loss.item()
    
    def retrieval_train(self,dataloader):
        # minloss = 1 << 32
        self.old_model = copy.deepcopy(self.model)
        missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),self.serial_id,missing_file=self.missing_file_name)
        for i, x in enumerate(dataloader):
            x_modalities = self.get_x_device(x)
            self.check_modalities(x_modalities,missing_maps)
            self.copyoldmodel(i)
            outcome = self.model_execute(x_modalities)
            loss = self.loss(outcome.t())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            # minloss = min(minloss,loss.item())
            self.checkMMParamsChange(i)
            self.lastmissing, self.currentmissing = self.currentmissing, False
        return loss.item()

    def copyoldmodel(self,batch_index):
        if batch_index <= 0 or not self.poverty_check:
            return
        if not self.lastmissing and self.currentmissing:
            self.old_model = copy.deepcopy(self.model)
    
    @torch.no_grad()
    def checkMissingParams(self,batch_index):
        if not self.poverty_check or batch_index <= 0 or self.checkTimes > 3:
            return
        if not self.lastmissing and self.currentmissing:
            self.checkTimes += 1
            old_state_dict = self.old_model.state_dict()
            cur_state_dict = self.model.state_dict()
            diff_maps = {}
            for key in old_state_dict:
                diff_maps[key] = abs((old_state_dict[key] - cur_state_dict[key]) / old_state_dict[key]).sum()
            if len(diff_maps) > 1:
                self.store_params_maps.append(diff_maps)
            # Do a deletion operation
            del old_state_dict,cur_state_dict
            
    @torch.no_grad()
    def checkMMParamsChange(self,batch_index):
        if not self.poverty_check or batch_index <= 0 or self.checkTimes > 3:
            return
        if not self.lastmissing and self.currentmissing:
            self.checkTimes += 1
            # cur_state_dict = self.model.state_dict()
            diff_maps = {}
            for name,param in self.model.named_parameters():
                state = self.optimizer.state[param]
                if 'exp_avg' not in state:
                    continue
                momt_t = state['exp_avg']
                ver_t = state['exp_avg_sq']
                beta1,beta2 = self.optimizer.defaults['betas']
                eps = self.optimizer.defaults['eps']
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                m_hat = momt_t / bias_correction1
                v_hat = ver_t / bias_correction2
                
                update_step = self.optimizer.param_groups[0]['lr'] * m_hat / (v_hat.sqrt() + eps)
                change_rate = torch.abs(update_step / param.data)
                diff_maps[name] = change_rate.mean()
                
            if len(diff_maps) > 1:
                self.store_params_maps.append(diff_maps)
    
    def explore_params_gap(self):
        self.important_layer_key = ''
        if len(self.store_params_maps) <= 1:
            return
        max_variance = 0
        
        for key in self.model.state_dict():
            avg_changed = 0
            for diff_maps in self.store_params_maps:
                if key not in diff_maps:
                    continue
                # avg_variance += diff_maps[key]  
                # diff_sum = diff_maps[key].sum()
                avg_changed += diff_maps[key]
            avg_changed /= len(self.store_params_maps)
            if max_variance < avg_changed:
                max_variance = avg_changed
                self.important_layer_key = key 
        # for key in self.model.state_dict():
        #     # avg_variance = 0
        #     current_key_max_value = torch.zeros_like(self.model.state_dict()[key]).sum()
        #     current_key_min_value = torch.zeros_like(self.model.state_dict()[key]) + (1 << 31)
        #     current_key_min_value = current_key_min_value.sum()
        #     for diff_maps in self.store_params_maps:
        #         # avg_variance += diff_maps[key]
        #         diff_sum = diff_maps[key].sum()
        #         current_key_max_value = max(current_key_max_value,diff_sum)
        #         current_key_min_value = min(current_key_min_value,diff_sum)
        #     norm_variance = current_key_min_value / current_key_max_value
        #     if max_variance < norm_variance:
        #         max_variance = norm_variance
        #         self.important_layer_key = key
        
    # pacfl cluster
    def get_U_mask(self):
        class_samples,sorted_X = self.calculate_data_ratio()
        self.calculate_U_mask(class_samples,sorted_X)
        return self.U_mask
        
    def calculate_data_ratio(self):
        # the sequence of train_data is messy
        train_data = loaderapi.DatasetGenerator[self.dataset](
            self.dataset,
            self.serial_id,
            self.dataset_dir,
            const.DataType[0],
            is_preload = self.is_data_preload
        )
        # train_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train = True)
        if not self.is_data_preload:
            X_values1,X_values2 = [data[0][0] for data in train_data],[data[0][1] for data in train_data]
            res = pretrain.after_clipprocessor((X_values1,X_values2))
            X_values = res['pixel_values']
        else:
            X_values = torch.stack([data[0][0] for data in train_data],dim=0)
        y_values = torch.stack([data[1] for data in train_data],dim=0)
        # sorted_indices = np.argsort(y_values)
        sorted_indices = torch.argsort(y_values)
        sorted_X = X_values[sorted_indices]
        sorted_y = y_values[sorted_indices]
        # unique_y,counts = np.unique(sorted_y,return_counts = True)
        unique_y,counts = torch.unique(sorted_y,return_counts = True)
        class_samples = {}
        for y,count in zip(unique_y,counts):
            class_samples[y] = count
        possess_class_nums = len(unique_y)
        base = 1 / possess_class_nums
        temp_ratio = {}
        
        for class_k in class_samples:
            proportion_k = class_samples[class_k]
            temp_ratio[class_k] = proportion_k
            if proportion_k >= (base+0.05):
                temp_ratio[class_k] = class_samples[class_k]
        
        sub_sum = sum(list(temp_ratio.values()))
        
        for class_k in temp_ratio.keys():
            temp_ratio[class_k] = (temp_ratio[class_k]/sub_sum)*self.budget
        
        round_ratio = self.round_to(list(temp_ratio.values()), self.budget)
        cnt = 0
        for class_k in temp_ratio.keys():
            temp_ratio[class_k] = round_ratio[cnt]
            cnt += 1
        self.train_count_ratio = temp_ratio 
        return class_samples,sorted_X
        
    def calculate_U_mask(self,class_samples,sorted_X):
        cnt = 0
        U_temp = []
        K = 0
        for class_k,samples in class_samples.items():
            local_label_data = sorted_X[cnt:cnt+samples]
            local_label_data = torch.Tensor(local_label_data)
            local_label_data = local_label_data.reshape(samples.item(),-1).T
            # if type(sorted_y[cnt:cnt+samples]) == torch.Tensor:
            #     local_labels = list(set(sorted_y[cnt:cnt+samples].numpy()))
            # else:
            #     local_labels = list(set(sorted_y[cnt:cnt+samples]))
            if self.partition == 'dirichlet':
                if class_k in self.train_count_ratio.keys():
                    K = self.train_count_ratio[class_k]
                else:
                    K = self.nbias
            
            if K > 0:
                U1_temp,sh1_temp,vh1_temp = np.linalg.svd(local_label_data,full_matrices=False)
                U1_temp = U1_temp / np.linalg.norm(U1_temp,ord = 2,axis = 0)
                U_temp.append(U1_temp[:,0:K])
            cnt += samples

        self.U_mask = np.concatenate(U_temp,axis=1)  
        # print(self.U_mask.shape)  
            
    def round_to(self,percents, budget=100):
        if not np.isclose(sum(percents), budget):
            raise ValueError
        n = len(percents)
        rounded = [int(x) for x in percents]
        up_count = budget - sum(rounded)
        errors = [(self.error_gen(percents[i], rounded[i] + 1) - self.error_gen(percents[i], rounded[i]), i) for i in range(n)]
        rank = sorted(errors)
        for i in range(up_count):
            rounded[rank[i][1]] += 1
        return rounded
    
    def error_gen(self,actual, rounded):
        divisor = np.sqrt(1.0 if actual < 1.0 else actual)
        return abs(rounded - actual) ** 2 / divisor
