import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F

class ImageToTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, hidden_size=512,num_heads = 8,num_layers = 2):
        super().__init__()
        
        # Load pre-trained ResNet (feature extractor)
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的池化层和全连接层
        self.fc_img = nn.Linear(resnet.fc.in_features, embed_size)  # 将特征映射到 Transformer 的嵌入维度
        
        # Transformer 解码器（只使用解码器部分）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 嵌入层和输出层
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions=None,cap_lengths = None,generate_lengths = 20):
        features = self.extract_image_features(images)  # (batch_size, num_patches, embed_size)
        
        if self.training and captions is not None:
            # 训练模式：使用真实的 captions 来训练
            embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
            embeddings = embeddings.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, embed_size) 供 Transformer 使用
            
            packed_embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, cap_lengths, batch_first=False, enforce_sorted=False
            )
            
            # 利用 cross-attention 在图像特征上生成文本
            features = features.permute(1, 0, 2)  # (num_patches, batch_size, embed_size)
            outputs = self.decoder(embeddings, features)  # (seq_len, batch_size, embed_size)
            
            outputs = outputs.permute(1, 0, 2)  # 转换为 (batch_size, seq_len, embed_size)
            outputs = self.fc(outputs)  # (batch_size, seq_len, vocab_size)
            return outputs
        else:
            # 推理模式：从图像生成 captions
            return self.generate_caption(features,generate_lengths)

    def extract_image_features(self, images):
        # 提取图像特征
        features = self.resnet(images)  # (batch_size, channels, h, w)
        features = features.flatten(2).permute(0, 2, 1)  # (batch_size, num_patches, channels)
        features = self.fc_img(features)  # (batch_size, num_patches, embed_size)
        return features
    
    def generate_caption(self, features, max_length=20):
        batch_size = features.size(0)
        features = features.permute(1, 0, 2)  # (num_patches, batch_size, embed_size)
        
        # Initial input is <START> token
        inputs = torch.full((1, batch_size), 49406, dtype=torch.long, device=features.device)
        
        captions = []
        for i in range(max_length):
            # Embed the input tokens
            token_embeddings = self.embed(inputs)  # (1, batch_size, embed_size)
            
            # Pass through the decoder
            outputs = self.decoder(token_embeddings, features)  # (1, batch_size, embed_size)
            logits = self.fc(outputs)  # (1, batch_size, vocab_size)
            
            # Add temperature to increase diversity
            temperature = 1.0
            logits = logits / temperature
            
            # Sample from the distribution instead of taking argmax
            probs = F.softmax(logits.squeeze(0), dim=-1)
            predicted = torch.multinomial(probs, 1).squeeze(1)  # (batch_size)
            
            # print(f"Step {i}: Predicted tokens: {predicted}")  # Debug print
            
            captions.append(predicted)
            
            # If all generated tokens are <END>, stop early
            if (predicted == 49407).all():
                break
            
            # Use the current generated token as the next input
            inputs = predicted.unsqueeze(0)  # (1, batch_size)
        
        return torch.stack(captions, dim=1)  # (batch_size, seq_len)

    def contrastive_loss(self, img_embeddings, txt_embeddings, temperature=0.07):
        """
        对比学习的损失函数。计算图像嵌入和文本嵌入之间的相似度，使用 InfoNCE loss。
        """
        batch_size = img_embeddings.size(0)
        
        # 归一化嵌入向量
        img_embeddings = F.normalize(img_embeddings, p=2, dim=1)
        txt_embeddings = F.normalize(txt_embeddings, p=2, dim=1)
        
        # 计算相似性矩阵 (batch_size, batch_size)
        similarity_matrix = torch.matmul(img_embeddings, txt_embeddings.T) / temperature
        
        # 生成标签 (真实的相似性应该是对角线上的，即每对 (i, i) 是正例)
        labels = torch.arange(batch_size, device=img_embeddings.device)
        
        # 使用交叉熵损失，来优化 InfoNCE loss
        loss_img_to_txt = F.cross_entropy(similarity_matrix, labels)
        loss_txt_to_img = F.cross_entropy(similarity_matrix.T, labels)
        
        # 总损失是两个方向的平均
        loss = (loss_img_to_txt + loss_txt_to_img) / 2
        return loss


class TextToImageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_heads=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=49407)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_size, hidden_size * 7 * 7)
        
        self.deconv1 = nn.ConvTranspose2d(hidden_size, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, text, lengths):
        batch_size, seq_len = text.size()
        
        # lengths = torch.as_tensor(lengths)
        # Create mask for padding
        # mask = torch.arange(seq_len, device=text.device)[None, :] >= lengths[:, None]
        mask = generate_mask(lengths,seq_len)
        # Embed text
        x = self.embed(text)  # (batch_size, seq_len, embed_size)
        
        # Pass through Transformer encoder
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        x = self.encoder(x, src_key_padding_mask=mask)
        
        # Use only the non-padding tokens for further processing
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)
        x = torch.stack([x[i, :length].mean(dim=0) for i, length in enumerate(lengths)])
        
        # Generate image
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)  # (batch_size, 512, 7, 7)
        
        x = F.relu(self.deconv1(x))  # (batch_size, 256, 14, 14)
        x = F.relu(self.deconv2(x))  # (batch_size, 128, 28, 28)
        x = F.relu(self.deconv3(x))  # (batch_size, 64, 56, 56)
        x = F.relu(self.deconv4(x))  # (batch_size, 32, 112, 112)
        x = torch.tanh(self.deconv5(x))  # (batch_size, 3, 224, 224)
        
        return x

def get_lengths(a)->list:
    result = []
    for row in a:
        indices = torch.where(row == 49407)[0]
            # 如果找到匹配的值，获取第一个匹配的索引
        first_occurrence = indices[0].item() if indices.numel() > 0 else len(a)
        if first_occurrence == -1:
            print(a)
        result.append(first_occurrence)
    return result

def generate_mask(lengths, max_token_length):
    # lengths: (bs,)
    # max_token_length: scalar
    # print(lengths)
    bs = lengths.size(0)
    # print(bs)
    # Create a range tensor (1D tensor with values [0, 1, 2, ..., max_token_length - 1])
    range_tensor = torch.arange(max_token_length, device=lengths.device).expand(bs, max_token_length)
    
    # Expand the lengths tensor to compare with the range tensor
    lengths_expanded = lengths.unsqueeze(1)  # (bs, 1)

    # Generate mask by comparing range_tensor with lengths_expanded
    mask = range_tensor < lengths_expanded  # (bs, max_token_length)
    
    return mask