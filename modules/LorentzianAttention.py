import math

import torch
import torch.nn as nn

from manifolds.lorentz import Lorentz


# from layers.RoPE import RotaryEmbedding


class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the
                output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise
                it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim].
                If set to 0, then it is a normal lorentz linear layer.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlin=None, merge=False):
        super(LorentzLinear, self).__init__()
        self.nonlin = nonlin
        self.manifold = Lorentz()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)

    def forward(self, x, bias=None):

        if self.nonlin is not None:
            x = self.nonlin(x)
        if not self.merge:
            x = self.weight(self.dropout(x))
        else:
            x = self.weight(self.dropout(x.flatten(-2)))
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - self.manifold.k) / \
                (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzSelfAttention(nn.Module):

    def __init__(self, model_dim, dropout=0.1, require_RoPE=False):
        super(LorentzSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.manifold = Lorentz()
        self.linear_keys = LorentzLinear(
            model_dim, self.model_dim, dropout=dropout)
        self.linear_values = LorentzLinear(
            model_dim, self.model_dim, dropout=dropout)
        self.linear_query = LorentzLinear(
            model_dim, self.model_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(model_dim)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.residual = LorentzLinear(self.model_dim, self.model_dim, dropout=dropout, bias=False)
        # self.RoPE = RotaryEmbedding(dim=self.model_dim)
        # self.require_RoPE = require_RoPE

    def forward(self, query, key, value, mask=None):

        # effective_index = torch.sum(~mask, dim=-1) - 1
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)

        # if self.require_RoPE:
        #     size = query.size()
        #     query = query.reshape(-1, self.model_dim)
        #     query = self.RoPE.rotate_queries_or_keys(query).reshape(size)
        #     key = key.reshape(-1, self.model_dim)
        #     key = self.RoPE.rotate_queries_or_keys(key).reshape(size)

        batch_size = key.size(0)

        def shape(x):
            """ Projection. """
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, 1, self.model_dim)
            return x.transpose(1, 2)

        key = shape(key)
        value = shape(value)
        query = shape(query)
        query_len = query.size(2)
        key_len = key.size(2)
        inf = -2 ** 32 + 1
        # Q * V, calculate coefficient matrix.
        qk = self.manifold.cinner(query, key)
        attn = (2 + 2 * qk) / self.scale + self.bias
        # keep coefficients, using user embeddings 1...s calculates latent embedding for next user.

        # remove invalid cascades
        if mask is not None:
            if key_len == query_len:
                # mask = mask.unsqueeze(2) | mask.unsqueeze(1)
                pad_mask = mask.unsqueeze(dim=1).expand(-1, key_len, -1)
                tri_mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool().to(pad_mask.device)
                # diagonal=1 means not keep diagonal elements
                mask = tri_mask + pad_mask
                mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
                # attn = attn.masked_fill(mask, 0)
                attn = attn.masked_fill(mask, inf)
            else:
                mask = mask.unsqueeze(1).unsqueeze(1)
                attn = attn.masked_fill(mask, inf)

        attn = self.softmax(attn)

        # calculate latent embeddings.
        latent_emb = self.manifold.mid_point(value, attn)
        latent_emb = latent_emb.transpose(1, 2).squeeze(2)

        return latent_emb
