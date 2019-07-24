# Author: Kyle Kastner
# Based on code by Thomas Wolf.
# All rights reserved. This source code is licensed under the MIT-style license.
import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
def extract_blocks(a, blocksize):
    # extract_blocks(a, (2, 2))
    # if 4D, assume N H W C
    # returns N H W C patch_H patch_W
    if len(a.shape) != 4:
        raise ValueError("Currently only support len(a.shape) == 4")
    N, H, W, C = a.shape
    a = a.permute(0, 3, 1, 2)
    b0, b1 = blocksize
    a = a.contiguous()
    return a.reshape(N, C, H // b0, b0, W // b1, b1).transpose(3,4).reshape(N, C, H // b0, W // b1, b0, b1).permute(0, 2, 3, 1, 4, 5).contiguous()

class TransformerPositionEmbed(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super(TransformerPositionEmbed, self).__init__()
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
    def forward(self, x):
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.position_embeddings(positions)
        return h

class TransformerTokenEmbed(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super(TransformerTokenEmbed, self).__init__()
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
    def forward(self, x):
        h = self.tokens_embeddings(x)
        return h

class TransformerCombinedEmbed(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout):
        super(TransformerCombinedEmbed, self).__init__()
        self.position = TransformerPositionEmbed(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout)
        self.token = TransformerTokenEmbed(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ht = self.token(x)
        hp = self.position(x).expand_as(ht)
        h = hp + ht
        h = self.dropout(h)
        return h

class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, bidirectional=False):
        super(Transformer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional

        self.attentions = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        self.layer_norms_1 = nn.ModuleList()
        self.layer_norms_2 = nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        h = x
        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        if self.bidirectional:
            attn_mask = torch.zeros_like(attn_mask)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class BlockTransformer(nn.Module):
    def __init__(self, config):
        super(BlockTransformer, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.transformer_initial_token_embed = TransformerTokenEmbed(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                                                     config.num_max_positions, config.num_heads, config.num_layers,
                                                                     config.dropout)


        self.transformer_embed_h = nn.ModuleList()
        self.transformer_embed_w = nn.ModuleList()
        self.transformer_h_f = nn.ModuleList()
        self.transformer_w_f = nn.ModuleList()
        self.transformer_i_proj = nn.ModuleList()
        self.transformer_wh_proj = nn.ModuleList()
        self.transformer_o_proj = nn.ModuleList()
        if self.config.forward_backward:
            self.transformer_h_b = nn.ModuleList()
            self.transformer_w_b = nn.ModuleList()
        for i in range(config.num_layers):
            self.transformer_embed_h.append(TransformerPositionEmbed(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                                        config.num_max_positions, config.num_heads, config.num_layers,
                                                        config.dropout))
            self.transformer_embed_w.append(TransformerPositionEmbed(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                                        config.num_max_positions, config.num_heads, config.num_layers,
                                                        config.dropout))
            self.transformer_h_f.append(Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                           config.num_max_positions, config.num_heads, config.num_layers,
                                           config.dropout, bidirectional=config.bidirectional))
            if self.config.forward_backward:
                self.transformer_h_b.append(Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                               config.num_max_positions, config.num_heads, config.num_layers,
                                               config.dropout, bidirectional=config.bidirectional))
            # add intermediate projection
            self.transformer_i_proj.append(nn.Linear(config.embed_dim * config.image_size[0] // config.patch_size[0], config.embed_dim))
            self.transformer_w_f.append(Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                           config.num_max_positions, config.num_heads, config.num_layers,
                                           config.dropout, bidirectional=config.bidirectional))
            if self.config.forward_backward:
                self.transformer_w_b.append(Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                               config.num_max_positions, config.num_heads, config.num_layers,
                                               config.dropout, bidirectional=config.bidirectional))
            # project it back to the size needed H * f_dim 
            self.transformer_wh_proj.append(nn.Linear(config.embed_dim, config.embed_dim * config.image_size[0] // config.patch_size[0]))
            self.transformer_o_proj.append(nn.Linear(config.embed_dim, config.embed_dim))
        self.reduce_proj = nn.Linear(config.embed_dim, config.out_dim)
        self.out_l1 = nn.Linear(config.image_size[0] // config.patch_size[0] * config.image_size[1] // config.patch_size[1] * config.out_dim, config.hidden_dim)
        self.out_l2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.out_dim)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        # Assume N H W 
        shp = x.shape
        # 2 step
        # first permute target axis to the back, then flatten so that "features" go in sequential blocks
        # then swap first and last axis
        x_h = x.permute(1, 0, 2).type(torch.LongTensor if self.config.device[:3] == "cpu" else torch.cuda.LongTensor)
        shp_h = x_h.shape
        x_h = x_h.reshape(shp_h[0], shp_h[1] * shp_h[2]).contiguous()
        """
        # x_w not currently used, but could be
        x_w = x.permute(2, 0, 1).type(torch.LongTensor if self.config.device[:3] == "cpu" else torch.cuda.LongTensor)
        shp_w = x_w.shape
        x_w = x_w.reshape(shp_w[0], shp_w[1] * shp_w[2]).contiguous()
        x_e_w = self.transformer_embed_w(x_w)
        """

        x_e_h = self.transformer_initial_token_embed(x_h)

        inp = x_e_h
        for i in range(self.config.num_layers):
            # from here on, it should be 1 tenet "layer"
            inp = inp + self.transformer_embed_h[i](inp)
            inp = self.dropout(inp)

            h_h = self.transformer_h_f[i](inp)
            f_dim = h_h.shape[-1]
            if self.config.forward_backward:
                h_h = self.transformer_h_b[i](torch.flip(h_h, [0]))
                h_h = torch.flip(h_h, [0])
            h_part = h_h.permute(2, 0, 1).reshape(f_dim, shp_h[0], shp_h[1], shp_h[2]).permute(2, 1, 3, 0).contiguous()

            # now need to swap axes and process h_part, being sure to push *all* the dims recurred over, into feature
            # end result is [W, N, H * f_dim]
            h_part_to_w = h_part.permute(0, 2, 1, 3).reshape(shp[0], shp[2], shp[1] * f_dim).permute(1, 0, 2).contiguous()
            h_part_to_w_proj = self.transformer_i_proj[i](h_part_to_w)
            h_part_to_w_proj = h_part_to_w_proj + self.transformer_embed_w[i](h_part_to_w_proj)
            h_part_to_w_proj = self.dropout(h_part_to_w_proj)
            h_w = self.transformer_w_f[i](h_part_to_w_proj)
            if self.config.forward_backward:
                h_w = self.transformer_w_b[i](torch.flip(h_w, [0]))
                h_w = torch.flip(h_w, [0])

            # proj back to H * f_dim, then reshape it all back to image-like
            h_w_proj = self.transformer_wh_proj[i](h_w)
            w_part = h_w_proj.permute(1, 0, 2).reshape(shp[0], shp[2], shp[1], f_dim).permute(0, 2, 1, 3).contiguous()

            #comb = torch.cat([h_part, w_part], dim=-1)
            comb = self.dropout(w_part)
            proj_comb = nn.functional.relu(self.transformer_o_proj[i](comb))
            proj_comb = self.dropout(proj_comb)
            proj_comb = proj_comb.permute(1, 3, 0, 2).reshape(shp[1], f_dim, shp[0] * shp[2]).permute(0, 2, 1).contiguous()
            inp = proj_comb
            # end of 1 tenet "layer"

        reduce_comb = nn.functional.relu(self.reduce_proj(proj_comb))
        p_dim = reduce_comb.shape[-1]
        reduce_comb = reduce_comb.permute(2, 0, 1).reshape(p_dim, shp[1], shp[0], shp[2]).permute(2, 1, 3, 0)
        flat = reduce_comb.reshape(shp[0], -1)
        # 2 layer final projection
        l1 = nn.functional.relu(self.out_l1(flat))
        l1 = self.dropout(l1)
        l2 = nn.functional.relu(self.out_l2(l1))
        l2 = self.dropout(l2)
        out = self.out_proj(l2)
        return out


config = AttrDict()
config.vocab = 256
config.image_size = (28, 28)
config.patch_size = (1, 1)
# whether each transformer layer has directional masking - bidirectional means no masking
config.bidirectional = True
# whether to have 1 transformer (forward) or 2 (forward and backward)
config.forward_backward = False
config.status_every_n_minibatches = 500
config.status_every_n_seconds = 30
config.delay_train_misclassification = 10
config.delay_valid_misclassification = 0
config.num_epochs = 100
config.random_seed = 1999
config.num_layers = 2
config.batch_size = 13
config.embed_dim = 40
config.hidden_dim = 100
#config.num_layers = 2
#config.batch_size = 200
#config.embed_dim = 100
#config.hidden_dim = 512
config.out_dim = 10
config.num_max_positions = 256
config.num_embeddings = config.patch_size[0] * config.patch_size[1] * config.vocab
config.num_heads = 10
config.dropout = 0.0
config.initializer_range = 0.02
config.lr = 2.5E-4
config.max_norm = 0.25
config.n_warmup = 1000
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.gradient_accumulation_steps = 4
config.log_dir = "./"
config.dataset_cache = "./dataset_cache"

model = BlockTransformer(config).to(config.device)

train = datasets.MNIST('./data', train=True, download=True)
train_data = train.train_data
train_labels = train.train_labels
train_data = train_data[:, :, :, None]

valid = datasets.MNIST('./data', train=False, download=True)
valid_data = valid.test_data
valid_labels = valid.test_labels
valid_data = valid_data[:, :, :, None]

train_data = extract_blocks(train_data, config.patch_size)
valid_data = extract_blocks(valid_data, config.patch_size)

shp = train_data.shape
patch_shape = shp[-3:]
train_data = train_data.reshape(shp[0], shp[1], shp[2], -1)

shp = valid_data.shape
valid_data = valid_data.reshape(shp[0], shp[1], shp[2], -1)

features = train_data.shape[-1]

random_state = np.random.RandomState(config.random_seed)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=config.lr)

def make_all_batch_indices(dataset, batch_size, random_state=None):
    all_indices = [i for i in range(len(dataset))]
    if random_state is not None:
        random_state.shuffle(all_indices)
    final_indices = []
    i = 0
    while True:
        final_indices.append(tuple(all_indices[i:i + batch_size]))
        if i + batch_size >= len(all_indices):
            # ragged last batch
            final_indices.append(tuple(all_indices[i:]))
            break
        i = i + batch_size
    return final_indices


def preprocess(minibatch):
    # make it into a single integer "feature"
    minibatch[minibatch > 20] = 255
    minibatch[minibatch <= 20] = 0
    for i in range(minibatch.shape[-1]):
        minibatch[:, :, :, i] += i * config.vocab

    minibatch = minibatch.sum(dim=-1)
    return minibatch

print("Start training")
train_start = time.time()
epoch_losses = []
for e in range(config.num_epochs):
    last_train_status_time = time.time()
    running_loss = 0
    model.train()
    for n, inds in enumerate(make_all_batch_indices(train_data, config.batch_size, random_state)):
        minibatch = preprocess(train_data[inds, ...]).to(config.device)
        lbls = train_labels[inds, ...].to(config.device)

        opt.zero_grad()
        out = model(minibatch)

        loss = loss_fn(out, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)

        opt.step()
        running_loss += loss.data.cpu().numpy()
        if len(epoch_losses) == 0:
            epoch_losses.append(running_loss)
        if (n + 1) % config.status_every_n_minibatches == 0:
            print("Minibatch {} in Epoch {}: average loss so far {}".format(n + 1, e, running_loss / (n + 1)))
            last_train_status_time = time.time()
        elif time.time() - last_train_status_time > config.status_every_n_seconds:
            print("Minibatch {} in Epoch {}: average loss so far {}".format(n + 1, e, running_loss / (n + 1)))
            last_train_status_time = time.time()
    print("Epoch {}: average loss {}".format(e, running_loss / (n + 1)))
    epoch_losses.append(running_loss / (n + 1))

    model.eval()
    with torch.no_grad():
        print("Calculating misclassification rates...")
        last_valid_status_time = time.time()
        train_pred = []
        train_true = []
        all_inds  = make_all_batch_indices(train_data, config.batch_size, random_state)
        for n, inds in enumerate(all_inds):
            if e < config.delay_train_misclassification:
                break
            minibatch = preprocess(train_data[inds, ...]).to(config.device)
            lbls = train_labels[inds, ...].to(config.device)
            out = model(minibatch)
            pred = torch.max(out, dim=-1)[1]
            train_pred.extend([p for p in pred.data.cpu().numpy().ravel()])
            train_true.extend([l for l in lbls.data.cpu().numpy().ravel()])
            if time.time() - last_valid_status_time > config.status_every_n_seconds:
                print("Calculating training misclassification rate, minibatch {} of {}".format(n + 1, len(all_inds)))
                last_valid_status_time = time.time()

        if e >= config.delay_train_misclassification:
            assert len(train_pred) == len(train_true)
            train_right_wrong = [1 if tp == tt else 0 for tp, tt in zip(train_pred, train_true)]
            train_accuracy = sum(train_right_wrong) / float(len(train_true))
            print("Epoch {} train accuracy: {}".format(e, train_accuracy))
            print("Epoch {} train error: {}".format(e, 1 - train_accuracy))
            print("")

        last_valid_status_time = time.time()
        valid_pred = []
        valid_true = []
        all_inds  = make_all_batch_indices(valid_data, config.batch_size, random_state)
        for n, inds in enumerate(all_inds):
            if e < config.delay_valid_misclassification:
                break
            minibatch = preprocess(valid_data[inds, ...]).to(config.device)
            lbls = valid_labels[inds, ...].to(config.device)
            out = model(minibatch)
            pred = torch.max(out, dim=-1)[1]
            valid_pred.extend([p for p in pred.data.cpu().numpy().ravel()])
            valid_true.extend([l for l in lbls.data.cpu().numpy().ravel()])
            if time.time() - last_valid_status_time > config.status_every_n_seconds:
                print("Calculating validation misclassification rate, minibatch {} of {}".format(n + 1, len(all_inds)))
                last_valid_status_time = time.time()

        if e >= config.delay_valid_misclassification:
            assert len(valid_pred) == len(valid_true)
            valid_right_wrong = [1 if vp == vt else 0 for vp, vt in zip(valid_pred, valid_true)]
            valid_accuracy = sum(valid_right_wrong) / float(len(valid_true))
            print("Epoch {} valid accuracy: {}".format(e, valid_accuracy))
            print("Epoch {} valid error: {}".format(e, 1 - valid_accuracy))
            print("")

"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2, 2)
for i in range(axarr.shape[0]):
   for j in range(axarr.shape[1]):
       axarr[i, j].imshow(patches[i, j, 0])
plt.savefig("tmp.png")
"""
