import torch
import torchtext
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .transformer import PIENet
from algorithm.normalized import l2_normalize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from algorithm.sim.distance import cosine_sim
import const.defaultp as dp


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        embed_dim=256,
        no_imgnorm=False,
    ):
        super(ImageEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.fc(images)
        # When using pre-extracted region features, add an extra MLP for embedding transformation
        features = self.mlp(images) + features

        if not self.no_imgnorm:
            features = l2_normalize(features, axis=-1)

        return features


class TextEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        embed_dim=256,
        no_txtnorm=False,
        use_bi_gru=False,
    ):
        super(TextEncoder, self).__init__()
        # self.embed_dim = embed_dim
        # Sentence embedding
        # self.rnn = nn.GRU(
        #     input_dim, embed_dim // 2, bidirectional=True, batch_first=True
        # )
        self.no_txtnorm = no_txtnorm
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(
            input_dim, embed_dim, batch_first=True, bidirectional=use_bi_gru
        )

    def init_weights(self, cache_dir=os.path.join("cache/")):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_emb, lengths):
        # torch.backends.cudnn.enabled = False
        lengths = lengths.cpu()
        # Embed word ids to vectors
        # Forward propagate RNNs
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_emb = x_emb[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(
            x_emb, sorted_lengths.data.tolist(), batch_first=True
        )
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb[inv_ix]

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:, :, : int(cap_emb.size(2) // 2)]
                + cap_emb[:, :, int(cap_emb.size(2) // 2) :]
            ) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2_normalize(cap_emb, axis=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.Tensor(cap_emb.size(0), pad_size, cap_emb.size(2))
            if torch.cuda.is_available():
                pad_emb = pad_emb.cuda()
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return cap_emb


class Aggregation_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Aggregation_regulator, self).__init__()

        self.rar_q_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim), nn.Tanh(), nn.Dropout(0.4)
        )
        self.rar_k_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim), nn.Tanh(), nn.Dropout(0.4)
        )
        self.rar_v_w = nn.Sequential(nn.Linear(sim_dim, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mid, hig):
        mid_k = self.rar_k_w(mid)
        hig_q = self.rar_q_w(hig)
        hig_q = hig_q.unsqueeze(1).repeat(1, mid_k.size(1), 1)

        weights = mid_k.mul(hig_q)
        weights = self.softmax(self.rar_v_w(weights).squeeze(2))

        new_hig = (weights.unsqueeze(2) * mid).sum(dim=1)
        new_hig = l2_normalize(new_hig, axis=-1)

        return new_hig


class Correpondence_regulator(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Correpondence_regulator, self).__init__()

        self.rcr_smooth_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim // 2), 
            nn.Tanh(), 
            nn.Linear(sim_dim // 2, 1)
        )
        self.rcr_matrix_w = nn.Sequential(
            nn.Linear(sim_dim, sim_dim * 2),
            nn.Tanh(),
            nn.Linear(sim_dim * 2, embed_dim),
        )
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, matrix, smooth):

        matrix = (self.tanh(self.rcr_matrix_w(x)) + matrix).clamp(min=-1, max=1)
        smooth = self.relu(self.rcr_smooth_w(x) + smooth)

        return matrix, smooth


class Alignment_vector(nn.Module):
    def __init__(self, sim_dim, embed_dim):
        super(Alignment_vector, self).__init__()

        self.sim_transform_w = nn.Linear(embed_dim, sim_dim)

    def forward(self, query, context, matrix, smooth):

        wcontext = cross_attention(query, context, matrix, smooth)
        sim_rep = torch.pow(torch.sub(query, wcontext), 2)
        sim_rep = l2_normalize(self.sim_transform_w(sim_rep), axis=-1)

        return sim_rep


def cross_attention(query, context, matrix, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    query = torch.mul(query, matrix)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2_normalize(attn, axis=-1)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=2)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    wcontext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    wcontext = torch.transpose(wcontext, 1, 2)
    wcontext = l2_normalize(wcontext, axis=-1)

    return wcontext


class SimilarityEncoder(nn.Module):
    def __init__(self, opt, embed_dim, sim_dim,device = 'cuda:0'):
        super(SimilarityEncoder, self).__init__()
        self.opt = opt
        self.embed_dim = embed_dim
        self.sim_dim = sim_dim
        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        if opt.regulator == "only_rar":
            rar_step, rcr_step, alv_step = opt.rar_step, 0, 1
        elif opt.regulator == "only_rcr":
            rar_step, rcr_step, alv_step = 0, opt.rcr_step, opt.rcr_step
        elif opt.regulator == "coop_rcar":
            rar_step, rcr_step, alv_step = (
                opt.rcar_step,
                opt.rcar_step - 1,
                opt.rcar_step,
            )
        else:
            raise ValueError("Something wrong with opt.self_regulator")

        self.rar_modules = nn.ModuleList(
            [Aggregation_regulator(sim_dim, embed_dim) for i in range(rar_step)]
        )
        self.rcr_modules = nn.ModuleList(
            [Correpondence_regulator(sim_dim, embed_dim) for j in range(rcr_step)]
        )
        self.alv_modules = nn.ModuleList(
            [Alignment_vector(sim_dim, embed_dim) for m in range(alv_step)]
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, img_emb, cap_emb, cap_lens):

        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            # Generate a cap_i_expand repr
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            #
            query = cap_i_expand if self.opt.attn_type == "t2i" else img_emb
            context = img_emb if self.opt.attn_type == "t2i" else cap_i_expand

            smooth = (
                self.opt.t2i_smooth
                if self.opt.attn_type == "t2i"
                else self.opt.i2t_smooth
            )
            matrix = torch.ones(self.embed_dim,device = self.device)

            # ------- several setting for RAR and RCR ---------#
            if self.opt.regulator == "only_rar":
                sim_mid = self.alv_modules[0](query, context, matrix, smooth)
                sim_hig = torch.mean(sim_mid, 1)
                for m, rar_module in enumerate(self.rar_modules):
                    sim_hig = rar_module(sim_mid, sim_hig)
                sim_i = self.sigmoid(self.sim_eval_w(sim_hig))

            elif self.opt.regulator == "only_rcr":
                for m, rcr_module in enumerate(self.rcr_modules):
                    sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                    matrix, smooth = rcr_module(sim_mid, matrix, smooth)
                wcontext = cross_attention(query, context, matrix, smooth)
                sim_i = cosine_sim(query, wcontext).mean(dim=1, keepdim=True)

            elif self.opt.regulator == "coop_rcar":
                for m, rar_module in enumerate(self.rar_modules):
                    sim_mid = self.alv_modules[m](query, context, matrix, smooth)
                    if m == 0:
                        sim_hig = torch.mean(sim_mid, 1)
                    if m < (self.opt.rcar_step - 1):
                        matrix, smooth = self.rcr_modules[m](sim_mid, matrix, smooth)
                    sim_hig = rar_module(sim_mid, sim_hig)
                sim_i = self.sigmoid(self.sim_eval_w(sim_hig))

            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (
        (ind >= lengths.unsqueeze(1))
        if set_pad_to_one
        else (ind < lengths.unsqueeze(1))
    )
    mask = mask.to(lengths.device)
    return mask


class SCANpp(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, embedding_dim = 512,device = 'cuda:0'):
        super(SCANpp, self).__init__()
        self.img_enc = ImageEncoder(
            # args['img_dim'],
            embed_dim= embedding_dim,
            # no_imgnorm=args['no_imgnorm']
        )
        self.txt_enc = TextEncoder(
            # input_dim=args['input_dim'],
            embed_dim= embedding_dim,
        )
        self.sim_enc = SimilarityEncoder(
            opt=dp.ModelArgs["SimilarityEncoder"],
            embed_dim = embedding_dim,
            sim_dim=256,
            device = device,
        )

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset
        # if torch.cuda.is_available():
        #     images = images.cuda()
        #     captions = captions.cuda()
        #     lengths = lengths.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward(self, x_modalities, kwargs=None):
        """One training step given images and captions."""
        # compute the embeddings
        images, captions, lengths, _ = x_modalities
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        # img_embs,cap_embs =
        return sims.permute(1, 0)
