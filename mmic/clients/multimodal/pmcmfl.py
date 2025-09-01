from .clientbase import MultmodalClientBase
from utils.multimodaldata.dataloader import load_multimodal_data
import const.constants as const
from utils.data import read_missing_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import os
from models.criterion.clip import ClipLossHalf
import numpy as np
import math
from utils.multimodaldata import api as mmapi
from torch.utils.data import DataLoader, SubsetRandomSampler

class PmcmFLClient(MultmodalClientBase):
    def __init__(self, args, id, train_samples, test_samples, serial_id, logkey, **kwargs):
        super().__init__(args, id, train_samples, test_samples, serial_id, logkey, **kwargs)
        self.__init_prototypes()
        self.clip_logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(self.device)
        self.img_loss_weight = 0.3
        self.text_loss_weight = 0.3
        self.fusion_loss_weight = 0.4
        self.loss_scaler = SimpleGradClipper
        self.kd_temperature = 0.5
        self.sample_nums_with_modal = {}
        self.ClipLossHalf = ClipLossHalf()
        
        
    def __init_prototypes(self):
        self.rep_dim = 64
        if self.dataset == 'food101':
            self.rep_dim = 512
        self.prototypes = {
            "img": torch.randn([self.num_classes, self.rep_dim], dtype=torch.float32),
            "text": torch.randn([self.num_classes, self.rep_dim], dtype=torch.float32),
            "fusion": torch.randn([self.num_classes, self.rep_dim], dtype=torch.float32),
            "target": torch.randn([self.num_classes, self.num_classes], dtype=torch.float32)
        }
        
        
    def load_train_dataset(self,data_type = const.DataType[0], shuffle = True, batch_size = 10,index = 'cluster'):
        if self.batch_size is not None:
            batch_size = self.batch_size
        train_data = mmapi.DatasetGenerator[self.dataset](self.dataset,index,self.dataset_dir,data_type=data_type,is_preload = self.is_data_preload)
        missing_maps = {}
        if data_type == const.DataType[0]:
            missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),idx=self.serial_id,missing_file=self.missing_file_name)
        random_indices = []
        if (data_type == const.DataType[0] and self.train_subrate < 1.0) or (data_type == const.DataType[1] and self.test_subrate < 1.0):
            sub_rate = self.train_subrate if data_type == const.DataType[0] else self.test_subrate
            sub_samples = int(len(train_data) * sub_rate)
            random_indices = np.random.choice(len(train_data), sub_samples, replace=False)
            shuffle = False
        return self.split_data_with_modal(train_data,random_indices,missing_maps,shuffle)
        # return DataLoader(train_data, batch_size=self.batch_size, drop_last=drop_last, shuffle=shuffle,collate_fn=collate_fn,sampler=sampler)
    
    def split_data_with_modal(self,train_data,random_indices,missing_maps,shuffle = True):
        completed_data = []
        modal1only_data = []
        modal2only_data = []
        if random_indices == []:
            random_indices = [idx for idx in range(len(train_data))]
        for idx in random_indices:
            if missing_maps is not None and idx in missing_maps:
                if missing_maps[idx] == 0:
                    modal2only_data.append(train_data[idx])
                else:
                    modal1only_data.append(train_data[idx])
            else:
                completed_data.append(train_data[idx])
        collate_fn = mmapi.CollateGenerator[self.dataset](self.serial_id,missing_maps)
        completed_loader = DataLoader(completed_data, batch_size=self.batch_size, shuffle=shuffle,collate_fn = collate_fn,drop_last = True)
        modal1only_loader = DataLoader(modal1only_data, batch_size=self.batch_size, shuffle=shuffle,collate_fn = collate_fn,drop_last = True)
        modal2only_loader = DataLoader(modal2only_data, batch_size=self.batch_size, shuffle=shuffle,collate_fn = collate_fn,drop_last = True)
        return completed_loader,modal1only_loader,modal2only_loader
        # return self.create_dataloader(completed_data,missing_maps),self.create_dataloader(modal1only_data,missing_maps),self.create_dataloader(modal2only_data,missing_maps)
    
    def train(self):
        for key in self.prototypes:
            self.prototypes[key] = self.prototypes[key].to(self.device)
        completed_loader,modal1only_loader,modal2only_loader =\
            self.load_train_dataset(data_type = const.DataType[0], shuffle = True, batch_size = self.batch_size,index = self.serial_id)
        # prototypeloader = self.load_prototype_data(data_type = const.DataType[0],shuffle = True, batch_size = self.batch_size,index = self.serial_id)
        # missing_maps = read_missing_data(os.path.join(self.dataset_dir,self.dataset),idx=self.serial_id,missing_file=self.missing_file_name)
        self.sample_nums_with_modal['completed'] = len(completed_loader)
        self.sample_nums_with_modal['modal1only'] = 0 if modal1only_loader is None else len(modal1only_loader)
        self.sample_nums_with_modal['modal2only'] = 0 if modal2only_loader is None else len(modal2only_loader)
        self.model.train()
        for e in range(self.local_epochs):
            self.train_with_data(completed_loader,modal1only_loader,modal2only_loader)
            # train with text-img modality
            # train with text only
            # train with img only
            # compute prototypes
        self.compute_prototypes(dataloader=completed_loader, device=self.device)
        for key in self.prototypes:
            self.prototypes[key] = self.prototypes[key].cpu()
        self.save_prototypes(ckpt_name="client{}_prototypes-{}.pth" .format(self.serial_id,self.logkey) )
    
    def train_with_data(self,completed_loader,modal1only_loader,modal2only_loader):
        # for i,data in enumerate(trainloader):
        #     x,y,indexs = data
        #     x_modalities = self.get_x_device(x)
        #     # self.tracker.track()
        #     y = y.to(self.device)
        #     # self.tracker.track()
        #     missing_indexs = self.check_modalities(indexs,missing_maps,x_modalities)
        self.train_single_loader(completed_loader,mode = -1)
        self.train_single_loader(modal1only_loader,mode = 0)
        self.train_single_loader(modal2only_loader,mode = 1)
    
    def save_prototypes(self, ckpt_name):
        torch.save(self.prototypes, os.path.join(self.dataset_dir, ckpt_name))
    
    def compute_prototypes(self, dataloader, device):
        '''
        dataloader = protoype dataloader
        '''
        # self.model.set_mode("vl")
        self.model.eval()
        self.model.to(device)
        
        img_rep_box = None
        text_rep_box = None
        fusion_rep_box = None
        target_box = None
        label_box = None

        with torch.no_grad():
            for x,y,_ in dataloader:
                # image = data["image"].to(device, non_blocking=True)
                # language_tokens = data["language_tokens"].to(device, non_blocking=True)
                # padding_mask = data["padding_mask"].to(device, non_blocking=True)
                x = self.get_x_device(x)
                y = label_smoothing(torch.eye(self.num_classes)[y])
                labels = y.to(self.device)
                logits, hidden_reps = self.model_execute(X = x,only_preds = False)
                img_rep = hidden_reps["img"].clone().detach()
                text_rep = hidden_reps["text"].clone().detach()
                fusion_rep = hidden_reps["fusion"].clone().detach()
                soft_logit = logits.clone().detach() / self.kd_temperature
                target = torch.nn.functional.softmax(soft_logit, dim=-1)
                
                img_rep_box = img_rep if img_rep_box is None else torch.cat([img_rep_box, img_rep], dim=0)
                text_rep_box = text_rep if text_rep_box is None else torch.cat([text_rep_box, text_rep], dim=0)
                fusion_rep_box = fusion_rep if fusion_rep_box is None else torch.cat([fusion_rep_box, fusion_rep], dim=0)
                target_box = target if target_box is None else torch.cat([target_box, target], dim=0)
                label_box = labels if label_box is None else torch.cat([label_box, labels], dim=0)

                # compute the sum of each prototype
        if label_box is None:
            return
        img_prototype_sum = torch.matmul(label_box.to(torch.float).T, img_rep_box)
        text_prototype_sum = torch.matmul(label_box.to(torch.float).T, text_rep_box)
        fusion_prototype_sum = torch.matmul(label_box.to(torch.float).T, fusion_rep_box)
        target_prototype_sum = torch.matmul(label_box.to(torch.float).T, target_box)
        total_weight_per_class = torch.sum(label_box.to(torch.float).T, dim=1, keepdim=True)

        # compute global prototype
        img_prototypes = img_prototype_sum / total_weight_per_class
        text_prototypes = text_prototype_sum / total_weight_per_class
        fusion_prototypes = fusion_prototype_sum / total_weight_per_class
        target_prototypes = target_prototype_sum / total_weight_per_class

        # fill NaN with 0
        img_prototypes = torch.where(torch.isnan(img_prototypes), torch.full_like(img_prototypes, 0.0), img_prototypes)
        text_prototypes = torch.where(torch.isnan(text_prototypes), torch.full_like(text_prototypes, 0.0), text_prototypes)
        fusion_prototypes = torch.where(torch.isnan(fusion_prototypes), torch.full_like(fusion_prototypes, 0.0), fusion_prototypes)
        target_prototypes = torch.where(torch.isnan(target_prototypes), torch.full_like(target_prototypes, 0.0), target_prototypes)

        # store prototype to cpu
        img_prototype_new = img_prototypes.detach().cpu()
        text_prototype_new = text_prototypes.detach().cpu()
        fusion_prototype_new = fusion_prototypes.detach().cpu()
        target_prototypes_new = target_prototypes.detach().cpu()

        # update dict
        zero_prototype = torch.zeros([1, self.rep_dim], device="cpu", dtype=torch.float32)
        for clas in range(self.num_classes):
            if not torch.equal(img_prototype_new[clas:clas+1, :], zero_prototype):
                self.prototypes["img"][clas:clas+1, :] = img_prototype_new[clas:clas+1, :]
            if not torch.equal(text_prototype_new[clas:clas+1, :], zero_prototype):
                self.prototypes["text"][clas:clas+1, :] = text_prototype_new[clas:clas+1, :]
            if not torch.equal(fusion_prototype_new[clas:clas+1, :], zero_prototype):
                self.prototypes["fusion"][clas:clas+1, :] = fusion_prototype_new[clas:clas+1, :]
        zero_target = torch.zeros([1, self.num_classes], device="cpu", dtype=torch.float32)
        for clas in range(self.num_classes):
            if not torch.equal(target_prototypes_new[clas:clas+1, :], zero_target):
                self.prototypes["target"][clas:clas+1, :] = target_prototypes_new[clas:clas+1, :]
        # self.model.cpu()
        # self.model.train()
    
    def train_single_loader(self,loader,mode = 0):
        if not loader:
            return
        self.model.train()
        for x,y,index in loader:
            # y_indexs = torch.argmax(y,dim = 1).detach()
            y_indexs = y.detach().to(torch.long)
            x = self.get_x_device(x)
            y = y.to(self.device)
            clip_loss_text,clip_loss_img = 0,0
            if mode == -1:
                logits,hidden_states = self.model_execute(X = x,only_preds = False, mode = mode)
                clip_loss_img = self.ClipLossHalf(
                    hidden_states['img'],self.prototypes['img'][y_indexs],
                    self.clip_logits_scale.exp(),
                )
                clip_loss_text = self.ClipLossHalf(
                    hidden_states['text'],self.prototypes['text'][y_indexs],
                    self.clip_logits_scale.exp(),
                )
                clip_loss_fusion = self.ClipLossHalf(
                    hidden_states["fusion"], self.prototypes["fusion"][y_indexs],
                    self.clip_logits_scale.exp()
                )
            # modal 0 only
            elif mode == 0:
                text_prototype = self.prototypes['text'][y_indexs]
                logits,hidden_states = self.model_execute(X = x,only_preds = False,prototype = text_prototype,mode = mode)
                clip_loss_img = self.ClipLossHalf(
                    hidden_states['img'],self.prototypes['img'][y_indexs],
                    self.clip_logits_scale.exp(),
                )
                clip_loss_fusion = self.ClipLossHalf(
                    hidden_states["fusion"], self.prototypes["fusion"][y_indexs],
                    self.clip_logits_scale.exp()
                )
            # modal 1 only
            elif mode == 1:
                img_prototype = self.prototypes['img'][y_indexs]
                logits,hidden_states = self.model_execute(X = x,only_preds = False,prototype = img_prototype,mode = mode)
                clip_loss_text = self.ClipLossHalf(
                    hidden_states['text'],self.prototypes['text'][y_indexs],
                    self.clip_logits_scale.exp(),
                )
                clip_loss_fusion = self.ClipLossHalf(
                    hidden_states["fusion"], self.prototypes["fusion"][y_indexs],
                    self.clip_logits_scale.exp()
                )
            loss = self.loss(input = logits,target = y)
            # clip_loss_img_value = np.nan
            # clip_loss_text_value = np.nan
            # clip_loss_fusion_value = np.nan
            # if mode == -1 or mode == 1:
            #     clip_loss_text_value = clip_loss_text.item()
            # if mode == -1 or mode == 0:
            #     clip_loss_img_value = clip_loss_img.item()
            # clip_loss_fusion_value = clip_loss_fusion.item()
            loss += (0 if clip_loss_img is None else clip_loss_img) * self.img_loss_weight
            loss += (0 if clip_loss_text is None else clip_loss_text) * self.text_loss_weight
            loss += (0 if clip_loss_fusion is None else clip_loss_fusion) * self.fusion_loss_weight
            if not math.isfinite(loss.item()):
                print('Loss ios {}, stopping training'.format(loss.item()))
            # grad_norm = self.loss_scaler(loss = loss,optimizer = self.optimizer,parameters = self.model.parameters())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def split_train_loader_with_modality(self,trainloader,missing_maps):
        completed_loader = []
        modal1only_loader = []
        modal2only_loader = []
        for x,y,index in trainloader:
            if missing_maps is None or index not in missing_maps:
                completed_loader.append((x,y,index))
            elif missing_maps[index] == 0:
                modal2only_loader.append((x,y,index))
            else:
                modal1only_loader.append(x,y,index)
        return self.create_dataloader(completed_loader,missing_maps),self.create_dataloader(modal1only_loader,missing_maps),self.create_dataloader(modal2only_loader,missing_maps)
    # def load_prototype_data(self,data_type = const.DataType[0],shuffle = True, batch_size = self.batch_size,index = self.serial_id):

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
    
    
class SimpleGradClipper:
    @staticmethod
    def get_grad_norm_(parameters):
        """计算梯度的范数"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        return total_norm.item()
    
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # 反向传播
        loss.backward(create_graph=create_graph)
        
        # 梯度裁剪
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = SimpleGradClipper.get_grad_norm_(parameters)
            # 更新优化器
            optimizer.step()
        else:
            norm = None
        
        return norm

def label_smoothing(one_hot_labels, smoothing=0.1):
    """
    对 one-hot 编码进行标签平滑。
    :param one_hot_labels: one-hot 编码，形状为 (batch_size, num_classes)
    :param smoothing: 平滑系数，控制软标签的平滑程度
    :return: 平滑后的软标签，形状为 (batch_size, num_classes)
    """
    num_classes = one_hot_labels.size(1)
    smoothed_labels = (1.0 - smoothing) * one_hot_labels + smoothing / num_classes
    return smoothed_labels