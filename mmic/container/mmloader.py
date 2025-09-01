from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
import ujson
from torchvision import models

class ImageTextSaver():
    def __init__(self,index,dataset_obj,device,save_path):
        ''' 
        Your input dataset_obj should be [img,text,label]
        '''
        # super().__init__()
        self.device = device
        self.data_len = len(dataset_obj)
        self.textList = []
        self.imgList = []
        self.labelList = []
        self.y_static = dict()
        for obj in dataset_obj:
            self.imgList.append(obj[1])
            self.textList.append(obj[3])
            self.labelList.append(obj[2])
            self.y_static[obj[2]] = self.y_static.get(obj[2],0) + 1
        ### img model
        self.imgmodel = models.mobilenet_v2(pretrained=True)
        self.imgmodel.classifier = self.imgmodel.classifier[:-1]
        # self.imgmodel = torch.hub.load(
        #     'pytorch/vision:v0.8.0', 'densenet201', pretrained=True
        # )
        self.imgmodel = self.imgmodel.to(self.device)
        self.imgmodel.eval()
        self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ### text model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.textmodel = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.textmodel.eval()
        self.textmodel.to(self.device)
        
        self.save_path = save_path
        self.index = index
        
    def __len__(self):
        return self.data_len
    
    def read_img(self,img_path):
        image  =Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(image)
        # input_dim:(3,224,224) output dim:(1,1280)
        with torch.no_grad():
            input_data = image_tensor.to(self.device).unsqueeze(dim=0)
            img_features = self.imgmodel(input_data).detach().cpu().numpy()
        return img_features
        # return image_tensor

    def read_text(self,text):
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.textmodel(**tokens)
            text_features = features.last_hidden_state.detach().cpu().numpy()[0]
        return text_features
    
    def save(self,save_type = 'train'):
        X,y = [],[]
        # y_static = {}
        for i in range(self.data_len):
            img_feature,text_feature = self.read_img(self.imgList[i]),self.read_text(self.textList[i])
            img_len,text_len = self.get_data_len(img_feature),self.get_data_len(text_feature)
            X.append([img_feature,text_feature,img_len,text_len,i])
            y.append(self.labelList[i])
            # y_static[self.labelList[i]] = y_static.get(self.labelList[i],0) + 1
            
        save_data = {'x':X,'y':y}
        # if save_type == 'train':
        save_files(self.index,os.path.join(self.save_path,save_type),save_data)

    def get_data_len(self,feature):
        if feature is not None: 
            if len(feature.shape) == 3: feature = feature[0]
            feature = torch.tensor(feature)
            feat_length = len(feature)
        else:
            # convert to shape
            pass
        return feat_length
    
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