import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.dropout import Dropout

class DenseNetBertMMModel(nn.Module):
    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output
        
    def __init__(self, dim_visual_repr=1280, dim_text_repr=768, dim_proj=100, num_class=2):
        super(DenseNetBertMMModel, self).__init__()
        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr

       
        imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
       
        #config = BertConfig()
        #textEncoder = BertModel(config).from_pretrained('bert-base-uncased')
        #config = AlbertConfig()
        #textEncoder = AlbertModel(config).from_pretrained('albert-base-v2')
        # config = ElectraConfig()
        # textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        # config = XLNetConfig()
        #textEncoder = XLNetModel(config).from_pretrained('xlnet-base-cased')


        # super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder, save_dir)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(2*dim_proj, num_class)
        

    def forward(self, x):
        imagex, textx,_,_ = x

        # Getting feature map (eqn. 1)
        # N, dim_visual_repr
        f_i = self.dropout(self.flatten_vis(imagex))
        e_i = self.dropout(textx[:,0,:])
        # Getting sentence representation (eqn. 2)
        # The authors used embedding associated with [CLS] to represent the whole sentence
        
        # Applying self-attention to f_i and e_i
        f_i_self_attn = self.apply_self_attention(f_i)  # N, dim_proj
        e_i_self_attn = self.apply_self_attention(e_i)  # N, dim_proj
        
        # Getting linear projections (eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(
            self.proj_visual(f_i_self_attn)))  # N, dim_proj
        e_i_tilde = F.relu(self.proj_text_bn(
            self.proj_text(e_i_self_attn)))  # N, dim_proj

        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        masked_v_i = torch.multiply(alpha_v_i, f_i_tilde)
        masked_e_i = torch.multiply(alpha_e_i, e_i_tilde)
        joint_repr = torch.cat((masked_v_i, masked_e_i),
                               dim=1)  # N, 2*dim_proj

        # Get class label prediction logits with final fully-connected layers
        class_res = self.cls_layer(self.dropout(F.relu(self.self_attn_bn(self.fc_as_self_attn(joint_repr)))))
        return class_res