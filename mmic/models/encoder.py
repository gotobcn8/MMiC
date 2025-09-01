import torch
import pickle
import torch.nn as nn
from .transformer import PIENet
import os
import torchtext
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from algorithm.normalized import l2_normalize

class EncoderText(nn.Module):
    def __init__(
        self,
        wemb_type="glove",
        word_dim=200,
        embed_dim=256,
        num_classes=4,
        scale=128,
        mlp_local=False,
    ):
        super(EncoderText, self).__init__()
        with open("repository/vocabs/coco_vocab.pkl", "rb") as fin:
            vocab = pickle.load(fin)
        word2idx = vocab["word2idx"]

        self.embed_dim = embed_dim

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)

        # Sentence embedding
        self.rnn = nn.GRU(
            word_dim, embed_dim // 2, bidirectional=True, batch_first=True
        )

        self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)
        if torch.cuda.is_available():
            self.pie_net = self.pie_net.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.class_fc = nn.Linear(embed_dim, num_classes)
        self.class_fc_2 = nn.Linear(embed_dim, 80)

        self.init_weights(wemb_type, word2idx, word_dim)
        self.is_train = True
        self.phase = ""
        self.scale = scale

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
            )

    def init_weights(
        self, wemb_type, word2idx, word_dim, cache_dir=os.path.join("cache/")
    ):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if "fasttext" == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif "glove" == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception("Unknown word embedding type: {}".format(wemb_type))
            assert (
                wemb.vectors.shape[1] == word_dim
            ), f"wemb.vectors.shape {wemb.vectors.shape}"

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace("-", "").replace(".", "").replace("'", "")
                    if "/" in word:
                        word = word.split("/")[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print(
                "Words: {}/{} found in vocabulary; {} words missing".format(
                    len(word2idx) - len(missing_words),
                    len(word2idx),
                    len(missing_words),
                )
            )

    def forward(self, x, lengths):
        # torch.backends.cudnn.enabled = False
        lengths = lengths.cpu()
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        # if torch.cuda.device_count() > 1:
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I.to(x.device)).squeeze(1)
        out = padded[0][torch.arange(padded[0].size(0)), lengths - 1]
        output = {}
        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        # print('1', out.device, wemb_out.device, pad_mask.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask.to(out.device))

        # return out
        out = l2_normalize(out)

        if self.mlp_local:
            out = self.head_proj(out)

        output["embedding"] = out

        return output


class StTextEncoder(nn.Module):
    def __init__(
        self,
        wemb_type="glove",
        word_dim=200,
        embed_dim=256,
        num_classes=4,
        scale=128,
        mlp_local=False,
    ):
        super(StTextEncoder, self).__init__()
        with open("repository/vocabs/coco_vocab.pkl", "rb") as fin:
            vocab = pickle.load(fin)
        word2idx = vocab["word2idx"]

        self.embed_dim = embed_dim
        self.vocab_size = len(word2idx)
        # Word embedding
        self.embed = nn.Embedding(self.vocab_size, word_dim)

        # Sentence embedding
        self.rnn = nn.GRU(
            word_dim, embed_dim // 2, bidirectional=True, batch_first=True
        )

        self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)
        if torch.cuda.is_available():
            self.pie_net = self.pie_net.cuda()

        self.relu = nn.ReLU(inplace=False)
        self.class_fc = nn.Linear(embed_dim, num_classes)
        self.class_fc_2 = nn.Linear(embed_dim, 80)

        self.init_weights(wemb_type, word2idx, word_dim)
        self.is_train = True
        self.phase = ""
        self.scale = scale

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
            )

    def init_weights(
        self, wemb_type, word2idx, word_dim, cache_dir=os.path.join("cache/")
    ):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if "fasttext" == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif "glove" == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception("Unknown word embedding type: {}".format(wemb_type))
            assert (
                wemb.vectors.shape[1] == word_dim
            ), f"wemb.vectors.shape {wemb.vectors.shape}"

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace("-", "").replace(".", "").replace("'", "")
                    if "/" in word:
                        word = word.split("/")[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print(
                "Words: {}/{} found in vocabulary; {} words missing".format(
                    len(word2idx) - len(missing_words),
                    len(word2idx),
                    len(missing_words),
                )
            )

    def forward(self, x, lengths):
        max_index = torch.max(x)
        min_index = torch.min(x)
        if max_index >= self.vocab_size or min_index < 0:
            return IndexError("vocab index is exceed range")
        lengths = lengths.cpu()
        # Embed word ids to vectors
        wemb_out = self.embed(x)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        # if torch.cuda.device_count() > 1:
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I.to(x.device)).squeeze(1)

        pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
        # print('1', out.device, wemb_out.device, pad_mask.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask.to(out.device))
        out = out * self.scale
        out = self.relu(out)

        if self.is_train:
            fc_weight_relu = self.relu(self.class_fc.weight)
            self.class_fc.weight.data = fc_weight_relu
            x = self.class_fc(out)

            fc_weight_relu2 = self.relu(self.class_fc_2.weight)
            self.class_fc_2.weight.data = fc_weight_relu2
            x2 = self.class_fc_2(out)

            return x, x2, fc_weight_relu, fc_weight_relu2

        if self.mlp_local:
            out = self.head_proj(out)

        out = F.normalize(out, p=2, dim=1)
        return out


class EncoderImage(nn.Module):
    def __init__(self, config, mlp_local):
        super(EncoderImage, self).__init__()

        embed_dim = config["embedding_dim"]

        # Backbone CNN
        self.cnn = getattr(models, config["cnn_type"])(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_dim)

        self.cnn.fc = nn.Sequential()

        self.pie_net = PIENet(1, cnn_dim, embed_dim, cnn_dim // 2)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True

        self.n_samples_inference = config.get("n_samples_inference", 0)

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
            )

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)

        output = {}
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)

        out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))

        if self.mlp_local:
            out = self.head_proj(out)

        out = l2_normalize(out)

        output["embedding"] = out

        return output


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (
        (ind >= lengths.unsqueeze(1))
        if set_pad_to_one
        else (ind < lengths.unsqueeze(1))
    )
    mask = mask.to(lengths.device)
    return mask

