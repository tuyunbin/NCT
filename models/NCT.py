import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from einops.layers.torch import Rearrange
from einops import rearrange


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.model.transformer_encoder.att_dim
        self.num_attention_heads = cfg.model.transformer_encoder.att_head
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    self.hidden_size,
                    self.all_head_size,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                    groups=self.all_head_size
                )),
                # ('gn', nn.GroupNorm(32, self.all_head_size)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))

        self.key = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                self.hidden_size,
                self.all_head_size,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
                groups=self.all_head_size
            )),
            # ('gn', nn.GroupNorm(32, self.all_head_size)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))

        self.value = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                self.hidden_size,
                self.all_head_size,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
                groups=self.all_head_size
            )),
            # ('gn', nn.GroupNorm(32, self.all_head_size)),
            ('rearrage', Rearrange('b c h w -> b (h w) c')),
        ]))

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores #+ attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        res = rearrange(query_states,('b c h w -> b (h w) c') )
        context_layer += res
        context_layer = self.layer_norm(context_layer)

        return context_layer


class WeightAverage(nn.Module):
    def __init__(self, cfg, R=3):
        super(WeightAverage, self).__init__()
        self.c_in = cfg.model.transformer_encoder.att_dim
        self.c_out = self.c_in // 2

        self.conv_theta = nn.Conv2d(self.c_in, self.c_out, 1)
        self.conv_phi = nn.Conv2d(self.c_in, self.c_out, 1)
        self.conv_g = nn.Conv2d(self.c_in, self.c_out, 1)
        self.CosSimLayer = nn.CosineSimilarity(dim=3)  # norm
        self.drop = nn.Dropout(0.1)
        self.conv_back = nn.Conv2d(self.c_out, self.c_in, 1)

        self.R = R

    def forward(self, x):
        """
        x: torch.Tensor(batch_size, channel, h, w)
        """

        batch_size, c, h, w = x.size()
        padded_x = F.pad(x, (1, 1, 1, 1), 'replicate')
        neighbor = F.unfold(padded_x, kernel_size=self.R,
                            dilation=1, stride=1)  # BS, C*R*R, H*W
        neighbor = neighbor.contiguous().view(batch_size, c, self.R, self.R, h, w)
        neighbor = neighbor.permute(0, 2, 3, 1, 4, 5)  # BS, R, R, c, h ,w
        neighbor = neighbor.reshape(batch_size * self.R * self.R, c, h, w)

        theta = self.conv_theta(x)  # BS, C', h, w
        phi = self.conv_phi(neighbor)   # BS*R*R, C', h, w
        g = self.conv_g(neighbor)     # BS*R*R, C', h, w

        phi = phi.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        phi = phi.permute(0, 4, 5, 3, 1, 2)  # BS, h, w, c, R, R
        theta = theta.permute(0, 2, 3, 1).contiguous().view(
            batch_size, h, w, self.c_out)   # BS, h, w, c
        theta_dim = theta

        cos_sim = self.CosSimLayer(
            phi, theta_dim[:, :, :, :, None, None])  # BS, h, w, R, R

        softmax_sim = F.softmax(cos_sim.contiguous().view(
            batch_size, h, w, -1), dim=3).contiguous().view_as(cos_sim)  # BS, h, w, R, R
        softmax_sim = self.drop(softmax_sim)
        g = g.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        g = g.permute(0, 4, 5, 1, 2, 3)  # BS, h, w, R, R, c_out

        weighted_g = g * softmax_sim[:, :, :, :, :, None]
        weighted_average = torch.sum(weighted_g.contiguous().view(
            batch_size, h, w, -1, self.c_out), dim=3)
        weight_average = weighted_average.permute(
            0, 3, 1, 2).contiguous()  # BS, c_out, h, w
        weight_average = self.conv_back(weight_average)

        ret = x + weight_average

        return ret


class CrossLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.model.transformer_encoder.att_dim
        self.self_attention = SelfAttention(cfg)
        # self.hidden_intermediate = Intermediate(cfg)
        # self.output = Output(cfg)  # linear + residual + layernorm

    def forward(self, q, k, v):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        # self_attention_mask = dec_mask.unsqueeze(1)
        # if diagonal_mask:  # mask subsequent words
        #     max_len = dec_mask.size(1)  # Lt
        #     self_attention_mask = self_attention_mask * \
        #         torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            q, k, v)  # (N, Lt, D)

        # 3, linear + add_norm
        # dec_enc_attention_output = self.hidden_intermediate(attention_output)
        # dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return attention_output  # (N, Lt, D)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.model.transformer_encoder.att_dim
        self.self_attention = WeightAverage(cfg)
        # self.hidden_intermediate = Intermediate(cfg)
        # self.output = Output(cfg)  # linear + residual + layernorm

    def forward(self, input_tensor):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        # self_attention_mask = dec_mask.unsqueeze(1)
        # if diagonal_mask:  # mask subsequent words
        #     max_len = dec_mask.size(1)  # Lt
        #     self_attention_mask = self_attention_mask * \
        #         torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            input_tensor)  # (N, Lt, D)

        # 3, linear + add_norm
        # dec_enc_attention_output = self.hidden_intermediate(attention_output)
        # dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return attention_output  # (N, Lt, D)


class ChangeDetectorDoubleAttDyn(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.transformer_encoder.input_dim
        self.dim = cfg.model.transformer_encoder.dim
        self.feat_dim = cfg.model.transformer_encoder.feat_dim

        self.att_dim = cfg.model.transformer_encoder.att_dim
        self.emb_dim = cfg.model.transformer_encoder.emb_dim

        self.num_hidden_layers = cfg.model.transformer_encoder.att_layer
        self.layer = nn.ModuleList([EncoderLayer(cfg)
                                    for _ in range(self.num_hidden_layers)])

        self.cross_layer = nn.ModuleList([CrossLayer(cfg)
                                    for _ in range(self.num_hidden_layers)])

        self.img = nn.Sequential(
            nn.Linear(self.feat_dim, self.att_dim),
            nn.LayerNorm(self.att_dim, eps=1e-6),
            nn.Dropout(0.1)
        )

        self.w_embedding = nn.Embedding(14, int(self.att_dim / 2))
        self.h_embedding = nn.Embedding(14, int(self.att_dim / 2))

        self.embed = nn.Sequential(
            nn.Conv2d(self.att_dim * 2, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.diff_embed = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.att_dim),
            # nn.LayerNorm(self.att_dim, eps=1e-6),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.diff_feat = nn.Sequential(
            nn.Linear(self.att_dim * 2, self.att_dim),
            # nn.LayerNorm(self.att_dim, eps=1e-6),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.embed_fc = nn.Sequential(
            nn.Linear(self.att_dim, self.emb_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        # self.apply(self.init_weights)


    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_1, input_2):
        batch_size, C, H, W = input_1.size()
        input_1 = input_1.view(batch_size, C, -1).permute(0, 2, 1)  # (128, 196, 1026)
        input_2 = input_2.view(batch_size, C, -1).permute(0, 2, 1)
        input_1 = self.img(input_1)  # (128,196, 512)
        input_2 = self.img(input_2)

        pos_w = torch.arange(W).cuda()
        pos_h = torch.arange(H).cuda()
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(W, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, H, 1)],
                                       dim=-1)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch_size, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)
        position_embedding = position_embedding.view(batch_size, self.att_dim, -1).permute(0, 2, 1)
        input_1 = input_1 + position_embedding
        input_2 = input_2 + position_embedding

        input_1 = input_1.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        input_2 = input_2.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        for layer_idx, layer_module in enumerate(self.layer):
            input_1 = layer_module(input_1)
            input_2 = layer_module(input_2)

        # input_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1)  # (128, 196, 1026)
        # input_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1)

        for layer_idx, layer_module in enumerate(self.cross_layer):
            input_1_no_change = layer_module(input_1, input_2, input_2)
            input_2_no_change = layer_module(input_2, input_1, input_1)

        input_1 = input_1.view(batch_size, self.att_dim, -1).permute(0, 2, 1)  # (128, 196, 1026)
        input_2 = input_2.view(batch_size, self.att_dim, -1).permute(0, 2, 1)

        input_diff_1 = input_1 - input_1_no_change
        input_diff_2 = input_2 - input_2_no_change

        input_diff = torch.cat([input_diff_1, input_diff_2], -1)
        input_diff = self.diff_embed(input_diff)

        input_1 = input_1.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        input_2 = input_2.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)

        input_diff = input_diff.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)
        # input_diff_2 = input_diff_2.permute(0, 2, 1).view(batch_size, self.att_dim, H, W)

        input_before = torch.cat([input_1, input_diff], 1)
        input_after = torch.cat([input_2, input_diff], 1)
        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = torch.sigmoid(self.att(embed_before))
        att_weight_after = torch.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_1)
        attended_1 = (input_1 * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_2)
        attended_2 = (input_2 * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended_1 = attended_2 - attended_1
        input_attended_2 = attended_1 - attended_2

        input_attended = torch.cat([input_attended_1, input_attended_2], -1)
        input_attended = self.diff_feat(input_attended)

        attended_1 = attended_1.unsqueeze(1)
        attended_2 = attended_2.unsqueeze(1)
        input_attended = input_attended.unsqueeze(1)
        output = torch.cat([attended_1, attended_2, input_attended], 1)
        output = self.embed_fc(output)

        return output, att_weight_before, att_weight_after


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
