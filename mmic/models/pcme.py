import sys

import torch.nn as nn
from transformers import BertModel, BertTokenizer
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from .encoder import EncoderText,EncoderImage,l2_normalize


class PCME(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, config):
        super(PCME, self).__init__()

        self.config = config
        self.embed_dim = config['embedding_dim']
        self.n_embeddings = 1
        if 'n_samples_inference' in config.keys():
            self.n_embeddings = config['n_samples_inference']
            
        self.img_enc = EncoderImage(config, config['mlp_local'])
        if 'not_bert' in config.keys() and config['not_bert']:
            self.txt_enc = EncoderText(
                embed_dim=config['embedding_dim'],
                word_dim=config['word_dim'],
            )
        else:
            self.txt_enc = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(768, self.embed_dim)

    def forward(self, images, sentences, captions_word, lengths):
        image_output = self.img_enc(images)
        if self.config['not_bert']:
            caption_output = self.txt_enc(sentences, lengths)  # sentences: [128,  seq_len], lengths: 128
        else:
            inputs = self.tokenizer(captions_word, padding=True, return_tensors='pt')
            for a in inputs:
                inputs[a] = inputs[a].cuda()
            caption_output = self.txt_enc(**inputs)
            caption_output = {'embedding': l2_normalize(self.linear(caption_output['last_hidden_state'][:, 0, :]))}  # [bsz, 768]

        return {
            'image_features': image_output['embedding'],
            'image_attentions': image_output.get('attention'),
            'image_residuals': image_output.get('residual'),
            'image_logsigma': image_output.get('logsigma'),
            'image_logsigma_att': image_output.get('uncertainty_attention'),
            'caption_features': caption_output['embedding'],
            'caption_attentions': caption_output.get('attention'),
            'caption_residuals': caption_output.get('residual'),
            'caption_logsigma': caption_output.get('logsigma'),
            'caption_logsigma_att': caption_output.get('uncertainty_attention'),
        }

    def image_forward(self, images):
        return self.img_enc(images)

    def text_forward(self, sentences, lengths):
        return self.txt_enc(sentences, lengths)
