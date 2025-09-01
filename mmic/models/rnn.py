import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class ProtoImageTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        img_dim: int,     # Image data input dim
        text_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        en_att: bool=False,     # Enable self attention or not
        att_name: str='fuse_base',       # Attention Name
        d_head: int=6,           # Head dim
        missing_contrast:bool = True
    ):
        super(ProtoImageTextClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.missing_contrast = missing_contrast
        # Projection head
        self.img_proj = nn.Sequential(
            # 128 -> 64
            nn.Linear(img_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            # 64 -> 64
            nn.Linear(d_hid, d_hid)
        )
            
        # RNN module
        self.text_rnn = nn.GRU(
            input_size=text_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        elif self.missing_contrast:
            self.shrink_layer = nn.Linear(2 * d_hid,d_hid)
            self.share_encode = nn.Linear(d_hid,d_hid)
            self.joint_contrast_layer_plus= nn.Linear(d_hid,d_hid)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        
        self.init_weight()
    
    def shared_encoder(self,x):
        return F.normalize(self.share_encode(x),dim = -1)
    
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def get_batch_representations(self,xv,xt):
        return [self.shared_encoder(xv),self.shared_encoder(xt)]
    
    def forward(self, X_modalities,**kwargs):
        prototype = kwargs.get('prototype',None)
        mode = kwargs.get('mode',-1)
        repr_maps = {}
        # 1. img proj
        x_img, x_text, len_i, len_t = X_modalities
        if mode is not None and mode == 1:
            x_img = prototype
            # using prototype represent 
        else:
            x_img = self.img_proj(x_img[:, 0, :])
        repr_maps['img'] = x_img.clone()
        # 2. Rnn forward
        if mode is not None and mode != 0:
            if len_t[0] != 0:
                x_text = pack_padded_sequence(
                    x_text,
                    len_t.cpu().numpy(), 
                    batch_first = True, 
                    enforce_sorted = False
                )
            self.text_rnn.flatten_parameters()
            x_text, _ = self.text_rnn(x_text)
            if len_t[0] != 0:
                x_text, _ = pad_packed_sequence(x_text, batch_first=True)
        else:
            x_text = prototype
        # 3. Attention
        if self.en_att:
            if self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                joint_repr = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            # 4. Average pooling
            if mode is None or mode != 0:
                x_text = torch.mean(x_text, axis=1)
            repr_maps['text'] = x_text.clone()
            joint_repr = torch.cat((x_img, x_text), dim=1)

        if not self.missing_contrast:
            preds = self.classifier(joint_repr)
            return preds, joint_repr
        
        joint_repr = self.shrink_layer(joint_repr)
        repr_maps['fusion'] = joint_repr.clone()
        joint_repr = self.shared_encoder(joint_repr)
        batch_reps = self.get_batch_representations(x_img,x_text)
        batch_reps.append(joint_repr)
        repr_maps['batch_reps'] = batch_reps
        output = self.joint_contrast_layer_plus(F.dropout(F.relu(joint_repr), p=0.0, training=self.training))
        # output += joint_repr

        # 4. MM embedding and predict
        preds = self.classifier(output)
        return preds, repr_maps

class ImageTextClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int,       # Number of classes 
        img_dim: int,     # Image data input dim
        text_dim: int,    # Text data input dim
        d_hid: int=64,          # Hidden Layer size
        en_att: bool=False,     # Enable self attention or not
        att_name: str='fuse_base',       # Attention Name
        d_head: int=6,           # Head dim
        missing_contrast:bool = True
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.1
        self.en_att = en_att
        self.att_name = att_name
        self.missing_contrast = missing_contrast
        # Projection head
        self.img_proj = nn.Sequential(
            # 128 -> 64
            nn.Linear(img_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            # 64 -> 64
            nn.Linear(d_hid, d_hid)
        )
            
        # RNN module
        self.text_rnn = nn.GRU(
            input_size=text_dim, 
            hidden_size=d_hid, 
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=False
        )

        # Self attention module
        if self.att_name == "fuse_base":
            self.fuse_att = FuseBaseSelfAttention(
                d_hid=d_hid,
                d_head=d_head
            )
        
        # classifier head
        if self.en_att and self.att_name == "fuse_base":
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*d_head, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        elif self.missing_contrast:
            self.shrink_layer = nn.Linear(2 * d_hid,d_hid)
            self.share_encode = nn.Linear(d_hid,d_hid)
            self.joint_contrast_layer_plus= nn.Linear(d_hid,d_hid)
            self.classifier = nn.Sequential(
                nn.Linear(d_hid, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        else:
            # classifier head
            self.classifier = nn.Sequential(
                nn.Linear(d_hid*2, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, num_classes)
            )
        
        self.init_weight()
    
    def shared_encoder(self,x):
        return F.normalize(self.share_encode(x),dim = -1)
    
    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def get_batch_representations(self,xv,xt):
        return [self.shared_encoder(xv),self.shared_encoder(xt)]
    
    def forward(self, X_modalities,**kwargs):
        # repr_maps = {}
        # 1. img proj
        x_img, x_text, len_i, len_t = X_modalities
        x_img = self.img_proj(x_img[:, 0, :])
        # repr_maps['img'] = x_img.clone()
        # 2. Rnn forward
        if len_t[0] != 0:
            x_text = pack_padded_sequence(
                x_text,
                len_t.cpu().numpy(), 
                batch_first = True, 
                enforce_sorted = False
            )
        self.text_rnn.flatten_parameters()
        x_text, _ = self.text_rnn(x_text)
        if len_t[0] != 0:
            x_text, _ = pad_packed_sequence(x_text, batch_first=True)
        # repr_maps['text'] = x_text.clone()
        # 3. Attention
        if self.en_att:
            if self.att_name == "fuse_base":
                # get attention output
                x_mm = torch.cat((x_img.unsqueeze(dim=1), x_text), dim=1)
                joint_repr = self.fuse_att(x_mm, len_i, len_t, 1)
        else:
            # 4. Average pooling
            x_text = torch.mean(x_text, axis=1)
            joint_repr = torch.cat((x_img, x_text), dim=1)

        if not self.missing_contrast:
            preds = self.classifier(joint_repr)
            return preds, joint_repr
        
        joint_repr = self.shrink_layer(joint_repr)
        # repr_maps['fusion'] = joint_repr.clone()
        joint_repr = self.shared_encoder(joint_repr)
        batch_reps = self.get_batch_representations(x_img,x_text)
        batch_reps.append(joint_repr)
        # repr_maps['batch_reps'] = batch_reps
        output = self.joint_contrast_layer_plus(F.dropout(F.relu(joint_repr), p=0.0, training=self.training))
        # output += joint_repr

        # 4. MM embedding and predict
        preds = self.classifier(output)
        return preds, batch_reps
    
    
class FuseBaseSelfAttention(nn.Module):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8421023
    def __init__(
        self, 
        d_hid:  int=64,
        d_head: int=4
    ):
        super().__init__()
        self.att_fc1 = nn.Linear(d_hid, 512)
        self.att_pool = nn.Tanh()
        self.att_fc2 = nn.Linear(512, d_head)

        self.d_hid = d_hid
        self.d_head = d_head

    def forward(
        self,
        x: Tensor,
        val_a=None,
        val_b=None,
        a_len=None
    ):
        att = self.att_pool(self.att_fc1(x))
        # att = self.att_fc2(att).squeeze(-1)
        att = self.att_fc2(att)
        att = att.transpose(1, 2)
        if val_a is not None:
            for idx in range(len(val_a)):
                # att[idx, :, val_a[idx]:a_len] = -1e5
                att[idx, :, 0:a_len] = -1e5
                att[idx, :, a_len+val_b[idx]:] = -1e5
        att = torch.softmax(att, dim=2)
        # x = torch.matmul(att, x).mean(axis=1)
        x = torch.matmul(att, x)
        x = x.reshape(x.shape[0], self.d_head*self.d_hid)
        return x