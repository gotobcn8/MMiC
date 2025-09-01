from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import os
import torch.nn as nn
import numpy as np
import ujson
# from torchvision import models
import torchvision.models as models
import time

# class ImageEncoder(nn.Module):
#     def __init__(self) -> None:
#         super(ImageEncoder,self).__init__()
#         self.imgmodel = models.resnet18(pretrained=True)
#         self.input_dim = self.imgmodel.fc.in_features
#         # self.embed_dim = 1280
        
        
#         # self.fc = nn.Linear(self.input_dim, self.embed_dim)
#         self.avgpool = self.imgmodel.avgpool
#         self.imgmodel.avgpool = nn.Sequential()
        
#     def forward(self,images):
#         out_7x7 = self.imgmodel(images).view(-1, self.input_dim, 7, 7)
#         pooled = self.+(out_7x7).view(-1, self.input_dim)
#         out = self.fc(pooled)
#         return out
    
class RetrievalSaver():
    def __init__(self,index,dataset_obj,device,save_path):
        ''' 
        Your input dataset_obj should be [img,text,label]
        '''
        # super().__init__()
        self.device = device
        self.data_len = len(dataset_obj)
        self.textList = []
        self.imgList = []
        for obj in dataset_obj:
            self.imgList.append(obj[0])
            self.textList.append(obj[1])
        ### img model
        self.init_imgmodel()
        ### text model
        self.init_textmodel()
        
        self.save_path = save_path
        self.index = index
    
    def __len__(self):
        return self.data_len
    
    def init_textmodel(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.textmodel = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.textmodel.eval()
        self.textmodel.to(self.device)
    
    def init_imgmodel(self):
        
        self.imgmodel = models.mobilenet_v2(pretrained=True)
        self.imgmodel.classifier = self.imgmodel.classifier[:-1]
        # self.imgmodel = torch.hub.load(
        #     'pytorch/vision:v0.8.0', 'densenet201', pretrained=True
        # )
        # self.imgmodel = ImageEncoder().to(self.device)
        self.imgmodel.to(self.device)
        self.imgmodel.eval()
        self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def read_img(self,img_path):
        img_path = os.path.join(self.save_path,'rawdata','flickr30k_images',img_path)
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image)
        with torch.no_grad():
            #inputdim:(3,224,224) -> (1,1280)
            input_data = image_tensor.to(self.device).unsqueeze(dim=0)
            img_features = self.imgmodel(input_data).detach().cpu().numpy()
        return img_features
        # return image_tensor

    def read_text(self,text):
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.textmodel(**tokens)
            text_features = features.last_hidden_state.detach().cpu().numpy()[0]
        return text_features,tokens.data['input_ids'].size(1)
    
    def save(self,save_type = 'train'):
        X = []
        # y_static = {}
        for i in range(self.data_len):
            img_feature,(text_feature,text_length) = self.read_img(self.imgList[i]),self.read_text(self.textList[i])
            X.append([img_feature,text_feature,text_length,i])
            # y_static[self.labelList[i]] = y_static.get(self.labelList[i],0) + 1
            
        save_data = {'x':X}
        # if save_type == 'train':
        save_files(self.index,os.path.join(self.save_path,save_type),save_data)
    
def save_files(index,save_path,save_data):
    if save_data is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, str(index) + '.npz'), 'wb') as f:
            np.savez_compressed(f, data=save_data)

# def save_files(idx,train_path,test_path,train_data,test_data,global_path,global_val_data,global_test_data):
#     if train_data is not None:
#         with open(os.path.join(train_path, str(idx) + '.npz'), 'wb') as f:
#             np.savez_compressed(f, data=train_data)
#     if test_data is not None:
#         with open(os.path.join(test_path ,str(idx) + '.npz'), 'wb') as f:
#             np.savez_compressed(f, data=test_data)
#     if global_val_data is not None:
#         with open(os.path.join(global_path,'val', str(idx)+'.npz'), 'wb') as f:
#             np.savez_compressed(f, data=global_val_data)
#     if global_test_data is not None:
#         with open(os.path.join(global_path,'test', str(idx)+'.npz'), 'wb') as f:
#             np.savez_compressed(f, data=global_test_data)
#     # with open(config_path, 'w') as f:
#     #     ujson.dump(config, f)

#     print("Finish generating dataset.\n")

def retrieval_encode_data(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions

    return img_embs, cap_embs, cap_lens