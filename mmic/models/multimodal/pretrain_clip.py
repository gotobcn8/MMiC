from transformers import CLIPModel
from torchvision.models.feature_extraction import create_feature_extractor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer,_make_causal_mask,_expand_mask
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import inspect
import traceback
import json

def get_pretrained_clip():
    home_path = os.getenv('HOME')
    clip_pretrain_name = 'clip-vit-base-patch32'
    remote_path = 'openai/' + clip_pretrain_name
    pretrain_save_path = os.path.join(home_path,'.pretrained',clip_pretrain_name)
    try:
        model = CLIPModel.from_pretrained(pretrain_save_path)
        # except OSError:
    except:
        model = CLIPModel.from_pretrained(remote_path)
            # model.save_pretrained(self.pretrain_save_path)
    return model

class PartialPretrainedClip(nn.Module):
    def __init__(self,num_classes = 10) -> None:
        super().__init__()
        self.text_attention_mask_key = 'text_attention_mask'
        self.text_casual_attention_mask_key = 'text_casual_attention_mask'
        self.img_attention_mask_key = 'visual_attention_mask'
        self.img_casual_attention_mask_key = 'visual_casual_attention_mask'
        self.text_layer_output = 'text_layer_norm2'
        self.img_layer_output = 'vision_layer_norm2'
        self.text_query_keys = (
            self.text_layer_output,
            self.text_attention_mask_key,
            self.text_casual_attention_mask_key
        )
        self.img_query_keys = (
            self.img_layer_output,
            self.img_attention_mask_key,
            self.img_casual_attention_mask_key
        )
        self.forward_output = {}
        self.forward_input = {}
        self.num_classes = num_classes
        '''
        At first was define in __method init_architecture, but couldn't link to device directly.
        '''
        # self.visual_projection = nn.Linear(in_features = 768,out_features = 512,bias = False)
        # self.text_projection = nn.Linear(in_features=512, out_features = 512, bias = False)
        self.text_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.visual_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
    
    @torch.no_grad()
    def pretrained_forward(self,inputs,basemodel):
        # inputs_dict = {
        #     'pixel_values':inputs[0],
        #     'input_ids':inputs[1],
        #     'attention_mask':inputs[2],
        # }
        attention_mask = inputs[2]
        text_hidden_state = basemodel.text_model(
            input_ids = inputs[1],
            attention_mask = inputs[2],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2]
        #  = text_res
        visual_hidden_state11 = basemodel.vision_model(
            pixel_values = inputs[0],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2][11]
        input_shape = inputs[1].shape
        text_hidden_state11 = text_hidden_state[11]
        causal_attention_mask = _make_causal_mask(input_shape, text_hidden_state[0].dtype, device=text_hidden_state[0].device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, text_hidden_state[0].dtype)
        # tmp = basemodel.text_model(**inputs_dict)
        del text_hidden_state
        return visual_hidden_state11,text_hidden_state11,attention_mask,causal_attention_mask
            
    # def init_architecture(self,basemodel):
    #     self.basemodel = basemodel
    #     self.basemodel.text_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.text_query_keys))
    #     self.basemodel.vision_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.img_query_keys))
    #     # Training last encoder layer in clip
    #     self.text_final_encoder_layer = copy.deepcopy(self.basemodel.text_model.encoder.layers[11])
    #     self.visual_final_encoder_layer = copy.deepcopy(self.basemodel.vision_model.encoder.layers[11])
    #     # Training final layernorm
    #     self.text_final_layer_norm = copy.deepcopy(self.basemodel.text_model.final_layer_norm)
    #     self.visual_final_layer_norm = copy.deepcopy(self.basemodel.vision_model.post_layernorm)
    #     # Define two projection
    #     self.text_projection = copy.deepcopy(self.basemodel.text_projection)
    #     self.visual_projection = copy.deepcopy(self.basemodel.visual_projection)
    
    def _init_architecture(self,basemodel):
        # basemodel.text_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.text_query_keys))
        # basemodel.vision_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.img_query_keys))
        # Training last encoder layer in clip
        self.text_final_encoder_layer = copy.deepcopy(basemodel.text_model.encoder.layers[11])
        self.visual_final_encoder_layer = copy.deepcopy(basemodel.vision_model.encoder.layers[11])
        # Training final layernorm
        self.text_final_layer_norm = copy.deepcopy(basemodel.text_model.final_layer_norm)
        self.visual_final_layer_norm = copy.deepcopy(basemodel.vision_model.post_layernorm)
        # Define two projection
        self.text_projection = copy.deepcopy(basemodel.text_projection)
        self.visual_projection = copy.deepcopy(basemodel.visual_projection)
        
    def forward(self,inputs,basemodel,kwargs = None):
        # (bs,len,768),        (bs,len,512)
        visual_layer_output,text_layer_output,attention_mask,casual_attention_mask = self.pretrained_forward(inputs,basemodel)
        # text_attention_mask,text_casual_attention_mask = self.forward_input[self.text_attention_mask_key],self.forward_input[self.text_casual_attention_mask_key]
        # img_attention_mask,img_casual_attention_mask = self.forward_input[self.img_attention_mask_key],self.forward_input[self.img_casual_attention_mask_key]
        # inspect.signature(self.text_final_encoder_layer.forward)
        text_layer_output.requires_grad = True
        visual_layer_output.requires_grad = True
        # print(text_layer_output.requires_grad,visual_layer_output.requires_grad)
        text_finallayer_output = self.text_final_encoder_layer(
            text_layer_output,
            attention_mask,
            casual_attention_mask
        )
        
        input_ids = inputs[1]
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        text_hidden_state = self.text_final_layer_norm(text_finallayer_output[0])
        text_pooled_output = text_hidden_state[
            torch.arange(text_hidden_state.shape[0], device=text_hidden_state.device),
            input_ids.to(dtype=torch.int, device=text_hidden_state.device).argmax(dim=-1),
        ]
        text_proj = self.text_projection(text_pooled_output)
        text_proj = self.text_pred_projection(text_proj)
        
        img_attention_mask,img_casual_attention_mask = None,None
        visual_finallayer_output = self.visual_final_encoder_layer(
            visual_layer_output,
            img_attention_mask,
            img_casual_attention_mask,
            # casual_attention_mask = None,
            # img_attention_mask,
            # img_casual_attention_mask
        )
        last_hidden_state = visual_finallayer_output[0]
        visual_pooled_output = last_hidden_state[:, 0, :]
        visual_pooled_output = self.visual_final_layer_norm(visual_pooled_output)
        visual_proj = self.visual_projection(visual_pooled_output)
        visual_proj = self.visual_pred_projection(visual_proj)
        preds = visual_proj * 0.5 + text_proj * 0.5
        # normalized features
        return preds


class ProtoPretrainedClip(nn.Module):
    def __init__(self,num_classes = 10) -> None:
        super().__init__()
        self.text_attention_mask_key = 'text_attention_mask'
        self.text_casual_attention_mask_key = 'text_casual_attention_mask'
        self.img_attention_mask_key = 'visual_attention_mask'
        self.img_casual_attention_mask_key = 'visual_casual_attention_mask'
        self.text_layer_output = 'text_layer_norm2'
        self.img_layer_output = 'vision_layer_norm2'
        self.text_query_keys = (
            self.text_layer_output,
            self.text_attention_mask_key,
            self.text_casual_attention_mask_key
        )
        self.img_query_keys = (
            self.img_layer_output,
            self.img_attention_mask_key,
            self.img_casual_attention_mask_key
        )
        self.forward_output = {}
        self.forward_input = {}
        self.num_classes = num_classes
        '''
        At first was define in __method init_architecture, but couldn't link to device directly.
        '''
        self.text_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.visual_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.joint_projection = nn.Linear(in_features=1024, out_features = 512, bias = False)
        
    @torch.no_grad()
    def pretrained_forward(self,inputs,basemodel):
        attention_mask = inputs[2]
        text_hidden_state = basemodel.text_model(
            input_ids = inputs[1],
            attention_mask = inputs[2],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2]
        #  = text_res
        visual_hidden_state11 = basemodel.vision_model(
            pixel_values = inputs[0],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2][11]
        input_shape = inputs[1].shape
        text_hidden_state11 = text_hidden_state[11]
        causal_attention_mask = _make_causal_mask(input_shape, text_hidden_state[0].dtype, device=text_hidden_state[0].device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, text_hidden_state[0].dtype)
        # tmp = basemodel.text_model(**inputs_dict)
        del text_hidden_state
        return visual_hidden_state11,text_hidden_state11,attention_mask,causal_attention_mask
    
    def _init_architecture(self,basemodel):
        self.text_final_encoder_layer = copy.deepcopy(basemodel.text_model.encoder.layers[11])
        self.visual_final_encoder_layer = copy.deepcopy(basemodel.vision_model.encoder.layers[11])
        # Training final layernorm
        self.text_final_layer_norm = copy.deepcopy(basemodel.text_model.final_layer_norm)
        self.visual_final_layer_norm = copy.deepcopy(basemodel.vision_model.post_layernorm)
        # Define two projection
        self.text_projection = copy.deepcopy(basemodel.text_projection)
        self.visual_projection = copy.deepcopy(basemodel.visual_projection)
        
    def forward(self,inputs,basemodel,**kwargs):
        # self.pretrained_forward(inputs,basemodel)
        # self.forward_input.clear()
        repr_maps = {}
        mode = kwargs.get('mode',-1)
        prototype = kwargs.get('prototype',None)
        visual_layer_output,text_layer_output,attention_mask,casual_attention_mask = self.pretrained_forward(inputs,basemodel)
        text_layer_output.requires_grad = True
        visual_layer_output.requires_grad = True
        if mode == 1 or mode == -1:
            text_finallayer_output = self.text_final_encoder_layer(
                text_layer_output,
                attention_mask,
                casual_attention_mask
            )
            
            input_ids = inputs[1]
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            text_hidden_state = self.text_final_layer_norm(text_finallayer_output[0])
            text_pooled_output = text_hidden_state[
                torch.arange(text_hidden_state.shape[0], device=text_hidden_state.device),
                input_ids.to(dtype=torch.int, device=text_hidden_state.device).argmax(dim=-1),
            ]
            text_proj = self.text_projection(text_pooled_output)
            repr_maps['text'] = text_proj.clone()
        else:
            text_proj = prototype
        
        if mode == 0 or mode == -1:
            img_attention_mask,img_casual_attention_mask = None,None
            visual_finallayer_output = self.visual_final_encoder_layer(
                visual_layer_output,
                img_attention_mask,
                img_casual_attention_mask,
            )
            last_hidden_state = visual_finallayer_output[0]
            visual_pooled_output = last_hidden_state[:, 0, :]
            visual_pooled_output = self.visual_final_layer_norm(visual_pooled_output)
            visual_proj = self.visual_projection(visual_pooled_output)  
            repr_maps['img'] = visual_proj.clone()
        else:
            visual_proj = prototype
        fusion_proj = self.joint_projection(torch.cat((text_proj,visual_proj),dim = 1))
        repr_maps['fusion'] = fusion_proj.clone()
        text_proj = self.text_pred_projection(text_proj)
        visual_proj = self.visual_pred_projection(visual_proj)
        preds = visual_proj * 0.5 + text_proj * 0.5
        # normalized features
        return preds, repr_maps

class GmcPartialClip(nn.Module):
    def __init__(self,num_classes = 10) -> None:
        super().__init__()
        self.text_attention_mask_key = 'text_attention_mask'
        self.text_casual_attention_mask_key = 'text_casual_attention_mask'
        self.img_attention_mask_key = 'visual_attention_mask'
        self.img_casual_attention_mask_key = 'visual_casual_attention_mask'
        self.text_layer_output = 'text_layer_norm2'
        self.img_layer_output = 'vision_layer_norm2'
        self.text_query_keys = (
            self.text_layer_output,
            self.text_attention_mask_key,
            self.text_casual_attention_mask_key
        )
        self.img_query_keys = (
            self.img_layer_output,
            self.img_attention_mask_key,
            self.img_casual_attention_mask_key
        )
        self.forward_output = {}
        self.forward_input = {}
        self.num_classes = num_classes
        '''
        At first was define in __method init_architecture, but couldn't link to device directly.
        '''
        # self.visual_projection = nn.Linear(in_features = 768,out_features = 512,bias = False)
        # self.text_projection = nn.Linear(in_features=512, out_features = 512, bias = False)
        self.text_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.visual_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.text_constrast_projection  = nn.Linear(in_features=512,out_features=256,bias = False)
        self.visual_constrast_projection  = nn.Linear(in_features=768,out_features=256,bias = False)
        self.joint_contrast_projection = nn.Linear(in_features=1280,out_features=512,bias=False)
        # self.joint_contrast_projection1 = nn.Linear(in_features=256,out_features=256,bias=False)
        # self.joint_contrast_projection2 = nn.Linear(in_features=256,out_features=256,bias=False)
        self.classifier = nn.Linear(in_features=128,out_features=self.num_classes,bias=False)
        
        # self.share_encode = nn.Linear(in_features = 256,out_features = 256)
        self.share_encode = nn.Linear(in_features = 512,out_features = 128)

    def shared_encoder(self,x):
        return F.normalize(self.share_encode(x),dim = -1)
    
    @torch.no_grad()
    def pretrained_forward(self,inputs,basemodel):
        # inputs_dict = {
        #     'pixel_values':inputs[0],
        #     'input_ids':inputs[1],
        #     'attention_mask':inputs[2],
        # }
        attention_mask = inputs[2]
        text_hidden_state = basemodel.text_model(
            input_ids = inputs[1],
            attention_mask = inputs[2],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2]
        #  = text_res
        visual_hidden_state11 = basemodel.vision_model(
            pixel_values = inputs[0],
            output_attentions = False,
            output_hidden_states = True,
            return_dict = False,
        )[2][11]
        input_shape = inputs[1].shape
        text_hidden_state11 = text_hidden_state[11]
        causal_attention_mask = _make_causal_mask(input_shape, text_hidden_state[0].dtype, device=text_hidden_state[0].device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, text_hidden_state[0].dtype)
        # tmp = basemodel.text_model(**inputs_dict)
        del text_hidden_state
        return visual_hidden_state11,text_hidden_state11,attention_mask,causal_attention_mask

    def _init_architecture(self,basemodel):
        # basemodel.text_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.text_query_keys))
        # basemodel.vision_model.encoder.layers[10].register_forward_hook(self.get_forward_info(self.img_query_keys))
        # Training last encoder layer in clip
        self.text_final_encoder_layer = copy.deepcopy(basemodel.text_model.encoder.layers[11])
        self.visual_final_encoder_layer = copy.deepcopy(basemodel.vision_model.encoder.layers[11])
        # Training final layernorm
        self.text_final_layer_norm = copy.deepcopy(basemodel.text_model.final_layer_norm)
        self.visual_final_layer_norm = copy.deepcopy(basemodel.vision_model.post_layernorm)
        # Define two projection
        self.text_projection = copy.deepcopy(basemodel.text_projection)
        self.visual_projection = copy.deepcopy(basemodel.visual_projection)
    
    def get_batch_representations(self,visual_batch,text_batch):
        # visual_batch = self.visual_constrast_projection(visual_batch)
        # text_batch = self.text_constrast_projection(text_batch)
        return [self.shared_encoder(visual_batch),self.shared_encoder(text_batch)]
        
    def forward(self,inputs,basemodel,kwargs = None):
        # self.pretrained_forward(inputs,basemodel)
        # self.forward_input.clear()
        visual_layer_output,text_layer_output,attention_mask,casual_attention_mask = self.pretrained_forward(inputs,basemodel)
        text_layer_output.requires_grad = True
        visual_layer_output.requires_grad = True
        # print(text_layer_output.requires_grad,visual_layer_output.requires_grad)
        text_finallayer_output = self.text_final_encoder_layer(
            text_layer_output,
            attention_mask,
            casual_attention_mask
        )
        
        input_ids = inputs[1]
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        text_hidden_state = self.text_final_layer_norm(text_finallayer_output[0])
        text_pooled_output = text_hidden_state[
            torch.arange(text_hidden_state.shape[0], device=text_hidden_state.device),
            input_ids.to(dtype=torch.int, device=text_hidden_state.device).argmax(dim=-1),
        ]
        text_proj = self.text_projection(text_pooled_output)
        # text_proj = self.text_pred_projection(text_proj)
        
        img_attention_mask, img_casual_attention_mask = None, None
        visual_finallayer_output = self.visual_final_encoder_layer(
            visual_layer_output,
            img_attention_mask,
            img_casual_attention_mask,
            # casual_attention_mask = None,
            # img_attention_mask,
            # img_casual_attention_mask
        )
        visual_hidden_state = visual_finallayer_output[0]
        visual_pooled_output = visual_hidden_state[:, 0, :]
        visual_pooled_output = self.visual_final_layer_norm(visual_pooled_output)
        visual_proj = self.visual_projection(visual_pooled_output)
        # visual_proj = self.visual_pred_projection(visual_proj)
        # preds = visual_proj * 0.5 + text_proj * 0.5
        # batch_representations = self.get_batch_representations(visual_pooled_output,text_pooled_output)
        batch_representations = self.get_batch_representations(visual_proj,text_proj)
        # 
        # joint_states = torch.cat((text_pooled_output,visual_pooled_output),dim = 1)
        joint_states = torch.cat((text_pooled_output,visual_pooled_output),dim = 1)
        # (,1280) -> (,512) -> (,128)
        joint_representation = self.shared_encoder(self.joint_contrast_projection(joint_states))
        batch_representations.append(joint_representation)
        # output = self.joint_contrast_projection2(F.dropout(F.relu(joint_representation), p=0.0, training=self.training))
        # (,128) -> (,num_classes)
        # output = self.classifier(F.dropout(F.relu(joint_representation), p=0.0, training=self.training))
        output = self.visual_pred_projection(visual_proj) + self.text_pred_projection(text_proj)

        # output = self.classifer()
        # output += joint_representation
        # normalized features
        return output,batch_representations

class UltraClipBase(nn.Module):
    def __init__(self,num_classes = 10) -> None:
        super().__init__()
        self.text_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)
        self.visual_pred_projection = nn.Linear(in_features=512, out_features = self.num_classes,bias = False)