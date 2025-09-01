import torch
from container.processer.pretrain import ClipProcessor,after_clipprocessor
from utils import vocab
import copy

class RetrievalCollateFn:
    '''
    This Collate function is used to control the missing modalities and missing map for missing modalities clients.
    missing_clients_map and missing_modality is mutex.\n
    But both missing_clients_map and missing_modality input, missing_clients_map priority.
    '''
    def __init__(self, index = None,missing_clients_map = dict(),missing_modality:list = []):
        self.missing_sets = missing_clients_map
        self.missing_modality = set(missing_modality)
        
    def __call__(self, batch):
        # 使用 self.param1 和 self.param2 处理 batch
        if self.missing_sets is not None:
            return self.missing_retrieval_collate_fn(batch, self.missing_sets)
        if self.missing_modality is not None:
            return self.collate_single_modal_flickr30k(batch,self.missing_modality)
        return self.retrieval_collate_fn(batch)
    
    def missing_retrieval_collate_fn(self,data,missing_sets):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        img_feats, cap_feats, lengths, indexs = zip(*data)
        max_cap_length = 0
        pad_cap_feats = []
        for cap_feat in cap_feats:
            max_cap_length = max(cap_feat.size(0),max_cap_length)
        img_feats = list(img_feats)
        for (i,cap_feat) in enumerate(cap_feats):
            cap_feat = pad_tensor(cap_feat,pad = max_cap_length)
            if indexs[i] in missing_sets:
                if missing_sets[indexs[i]] == 0:
                    img_feats[i] = torch.zeros(img_feats[i].size())
                elif missing_sets[indexs[i]] == 1:
                    cap_feat = torch.zeros(cap_feat.size())
            pad_cap_feats.append(cap_feat)
        # Merge x
        img_feats = torch.stack(img_feats, 0)
        pad_cap_feats = torch.stack(pad_cap_feats,0)
        lengths = torch.LongTensor(lengths)
        # Here need to be change to align length
        return img_feats, pad_cap_feats, lengths, indexs

    def retrieval_collate_fn(self,data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        img_feats, cap_feats, lengths, indexs = zip(*data)
        max_cap_length = 0
        pad_cap_feats = []
        for cap_feat in cap_feats:
            max_cap_length = max(cap_feat.size(0),max_cap_length)
        
        for cap_feat in cap_feats:
            pad_cap_feats.append(pad_tensor(cap_feat,pad = max_cap_length))
        # Merge x
        img_feats = torch.stack(img_feats, 0)
        pad_cap_feats = torch.stack(pad_cap_feats,0)
        lengths = torch.LongTensor(lengths)
        # Here need to be change to align length
        return img_feats, pad_cap_feats, lengths, indexs

    def collate_single_modal_flickr30k(self,data,modality = set([0])):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        img_feats, cap_feats, lengths, indexs = zip(*data)
        max_cap_length = 0
        pad_cap_feats = []
        for cap_feat in cap_feats:
            max_cap_length = max(cap_feat.size(0),max_cap_length)
        img_feats = list(img_feats)
        for (i,cap_feat) in enumerate(cap_feats):
            cap_feat = pad_tensor(cap_feat,pad = max_cap_length)
            if 0 in modality:
                img_feats[i] = torch.zeros(img_feats[i].size())
            if 1 in modality:
                cap_feat = torch.zeros(cap_feat.size())
            pad_cap_feats.append(cap_feat)
        # Merge x
        img_feats = torch.stack(img_feats, 0)
        pad_cap_feats = torch.stack(pad_cap_feats,0)
        lengths = torch.LongTensor(lengths)
        # Here need to be change to align length
        return img_feats, pad_cap_feats, lengths, indexs

class ClassifierCollateFn:
    def __init__(self, index = None,missing_clients_map = None,missing_modality:list = None):
        self.missing_sets = missing_clients_map
        self.missing_modality = missing_modality
        
    def __call__(self, batch):
        # 使用 self.param1 和 self.param2 处理 batch
        if self.missing_sets is not None:
            return self.classifier_collate_missing_modal(batch, self.missing_sets)
        if self.missing_modality is not None:
            return self.classifier_collate_single_modal(batch,set(self.missing_modality))
        return self.collate_mm_fn_padd(batch)

    def classifier_collate_missing_modal(self,batch,missing_sets):
        # find longest sequence
        if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
        if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
        
        # pad according to max_len
        x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
        indexs = list()
        for i,idx in enumerate(batch):
            batch_x = batch[i][0]
            xa = pad_tensor(batch_x[0], pad=max_a_len)
            xb = pad_tensor(batch_x[1], pad=max_b_len)
            index = batch[i][0][-1]
            if index in missing_sets:
                if missing_sets[index] == 0:
                    xa = torch.zeros(xa.size())
                elif missing_sets[index] == 1:
                    xb = torch.zeros(xb.size())
            x_a.append(xa)
            x_b.append(xb)
            len_a.append(batch_x[2])
            len_b.append(batch_x[3])
            # len_a.append(torch.tensor(batch[idx][0][2]))
            # len_b.append(torch.tensor(batch[idx][0][3])
            # )
            indexs.append(batch_x[4])
            ys.append(batch[i][-1])
        # stack all
        x_a = torch.stack(x_a, dim=0)
        x_b = torch.stack(x_b, dim=0)
        # len_a = torch.stack(len_a, dim=0)
        len_a = torch.Tensor(len_a)
        len_b = torch.Tensor(len_b)
        # len_b = torch.stack(len_b, dim=0)
        ys = torch.stack(ys, dim=0)
        return [x_a, x_b, len_a, len_b],ys,indexs

    def classifier_collate_single_modal(self,batch,modality = set([0])):
        # find longest sequence
        if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
        if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
        
        # pad according to max_len
        x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
        indexs = list()
        for i,idx in enumerate(batch):
            batch_x = batch[i][0]
            xa = pad_tensor(batch_x[0], pad=max_a_len)
            xb = pad_tensor(batch_x[1], pad=max_b_len)
            if 0 in modality:
                xa = torch.zeros(xa.size())
            if 1 in modality:
                xb = torch.zeros(xb.size())
            x_a.append(xa)
            x_b.append(xb)
            if len(batch_x) > 2:
                len_a.append(batch_x[2])
            if len(batch_x) > 3:
                len_b.append(batch_x[3])
            # len_a.append(torch.tensor(batch[idx][0][2]))
            # len_b.append(torch.tensor(batch[idx][0][3]))
            indexs.append(batch[i][4])
            ys.append(batch[i][-1])
        
        # stack all
        x_a = torch.stack(x_a, dim=0)
        x_b = torch.stack(x_b, dim=0)
        # len_a = torch.stack(len_a, dim=0)
        len_a = torch.Tensor(len_a)
        len_b = torch.Tensor(len_b)
        # len_b = torch.stack(len_b, dim=0)
        ys = torch.stack(ys, dim=0)
        return [x_a, x_b, len_a, len_b], ys, indexs

    def collate_mm_fn_padd(self,batch):
        # find longest sequence
        if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
        if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
        
        # pad according to max_len
        x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
        indexs = list()
        for idx in range(len(batch)):
            batch_x = batch[idx][0]
            x_a.append(pad_tensor(batch_x[0], pad=max_a_len))
            x_b.append(pad_tensor(batch_x[1], pad=max_b_len))
            
            len_a.append(batch_x[2])
            len_b.append(batch_x[3])
            # indexs.append(batch_x[4])
            indexs.append(batch_x[-1])
            ys.append(batch[idx][-1])
        
        # stack all
        x_a = torch.stack(x_a, dim=0)
        x_b = torch.stack(x_b, dim=0)
        # len_a = torch.stack(len_a, dim=0)
        len_a = torch.Tensor(len_a)
        len_b = torch.Tensor(len_b)
        # len_b = torch.stack(len_b, dim=0)
        
        ys = torch.stack(ys, dim=0)
        return [x_a, x_b, len_a, len_b], ys, indexs


class RawClassifierCollate():
    def __init__(self, index = None,missing_clients_map:dict = None,missing_modality:list = None,t2imodel = None,i2tmodel=None,device=None):
        self.missing_sets = missing_clients_map
        self.missing_modality = missing_modality
        self.t2i_model = t2imodel
        self.i2t_model = i2tmodel
        self.device = device
    def __call__(self, batch):
        # 使用 self.param1 和 self.param2 处理 batch
        if self.missing_sets is not None:
            return self.classifier_collate_missing_modal(batch, self.missing_sets)
        if self.missing_modality is not None:
            return self.classifier_collate_single_modal(batch,set(self.missing_modality))
        return self.collate_mm_fn_padd(batch)

    def t2imodelgen(self,text,lengths):
        generated_images = self.t2i_model(text,lengths)
        return generated_images
    
    def i2tmodelgen(self,img,max_length):
        # text = text.to(device)
        text = self.i2t_model(images = img,generate_lengths = max_length)
        return text

    def classifier_collate_missing_modal(self,batch,missing_sets):
        # find longest sequence
        # if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
        # if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
        
        # pad according to max_len
        # x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
        mx,label,index = zip(*batch)
        ma,mb = zip(*mx)
        # outputs = ClipProcessor(text=mb,images = ma,return_tensors = 'pt', padding=True)
        outputs = after_clipprocessor((ma,mb))
        xa,xb,xc = outputs['pixel_values'],outputs['input_ids'],outputs['attention_mask']
        if xb.size(1) > 77:
            xb = xb[:,:77]
            print('77 errors existed')
        for i in range(len(batch)):
            if index[i] not in missing_sets:
                continue
            missing_modal = missing_sets[index[i]]
            if missing_modal == 0:
                xa[i].zero_()
            elif missing_modal == 1:
                xb[i][1:-1] = 0
                if xb.max() > 50000:
                    print('is max:',xb)    
        label = torch.stack(label, dim=0)
        return [xa,xb,xc],label,index

    def classifier_collate_single_modal(self,batch,modality = set([0])):
        # find longest sequence
        if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
        if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
        
        # pad according to max_len
        x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
        for i,idx in enumerate(batch):
            batch_x = batch[i][0]
            xa = pad_tensor(batch_x[0], pad=max_a_len)
            xb = pad_tensor(batch_x[1], pad=max_b_len)
            if 0 in modality:
                xa = torch.zeros(xa.size())
            if 1 in modality:
                xb = torch.zeros(xb.size())
            x_a.append(xa)
            x_b.append(xb)
            if len(batch_x) > 2:
                len_a.append(batch_x[2])
            if len(batch_x) > 3:
                len_b.append(batch_x[3])
            # len_a.append(torch.tensor(batch[idx][0][2]))
            # len_b.append(torch.tensor(batch[idx][0][3]))

            ys.append(batch[i][-1])
        
        # stack all
        x_a = torch.stack(x_a, dim=0)
        x_b = torch.stack(x_b, dim=0)
        # len_a = torch.stack(len_a, dim=0)
        len_a = torch.Tensor(len_a)
        len_b = torch.Tensor(len_b)
        # len_b = torch.stack(len_b, dim=0)
        ys = torch.stack(ys, dim=0)
        return [x_a, x_b, len_a, len_b], ys

    def collate_mm_fn_padd(self,batch):
        # find longest sequence
        mx,label,index = zip(*batch)
        ma,mb = zip(*mx)
        inputs = after_clipprocessor((ma,mb))
        ys = torch.stack(label, dim=0) 
        # index = torch.stack(index,dim=0)
        return [inputs['pixel_values'],inputs['input_ids'],inputs['attention_mask']],ys,index
    
def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)
