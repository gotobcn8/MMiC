import torch.nn as nn
import torch

class GradsExplorer(nn.Module):
    def __init__(self, infeatures = 30, out_features = 30):
        super(GradsExplorer,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(30,128),
        )
        self.layer2 = nn.Linear(128,256)
        self.fc = nn.Linear(256,out_features)
        
    def forward(self,distribution_gaps):
        # x1,x2 = distribution_gaps
        x = torch.concat(distribution_gaps,dim = 0)
        x = torch.relu(self.layer1(x))  # ReLU 激活
        x = torch.relu(self.layer2(x))  # ReLU 激活
        x = self.fc(x)            
        return x

class InputDistributionTracker:
    def __init__(self, embedding_dim, decay=0.9):
        self.decay = decay
        self.ema_embedding = torch.zeros(embedding_dim)

    def update(self, batch_embedding):
        # 输入是当前 batch 的 mean embedding 向量
        self.ema_embedding = self.decay * self.ema_embedding + (1 - self.decay) * batch_embedding
        return self.ema_embedding.clone()  # 返回一个新的 tensor 防止 inplace 问题

class DistributionRNN(nn.Module):
    def __init__(self, image_input,text_input, hidden_dim, output_dim, num_layers=1):
        super(DistributionRNN, self).__init__()
        self.text_rnn = nn.GRU(
            input_size = text_input,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )
        self.image_rnn = nn.GRU(
            input_size = image_input,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, modal1, modal2):
        """
        input_seq: Tensor of shape [1, T, embedding_dim]
        Returns: predicted similarity vector of shape [output_dim]
        """
        rnn_out, _ = self.text_rnn(modal2)  # output shape: [1, T, hidden_dim]
        text_last_hidden = rnn_out[:, -1, :]   # take the last step's hidden state
        # out = self.fc(last_hidden)        # shape: [1, output_dim]
        
        image_rnn_out,_ = self.image_rnn(modal1)
        image_last_hidden = image_rnn_out[:,-1,:]
        out = text_last_hidden * 0.5 + image_last_hidden * 0.5
        out = out.squeeze(0)
        out = self.fc(out)
        return out.mean(dim=0)
