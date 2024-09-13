import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.conv1d = nn.Conv1d(in_channels=d_hid, out_channels=d_hid, kernel_size=5)
        self.maxpool = nn.MaxPool1d(3, stride=2)

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x) 
        x = self.w_2(x)
     
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x

def update_attn_weights(attn_weights, attn_multiplier):
    if attn_multiplier is not None:
        attn_weights = attn_weights * attn_multiplier[..., None]
        attn_weights = attn_weights / attn_weights.sum(1, keepdim=True)
    return attn_weights


class SelfonlyGradients(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attn_logits):
        return attn_logits

    @staticmethod
    def backward(ctx, grads):
        grads = torch.diagonal(grads, dim1=0, dim2=1)
        
        grads = torch.diag_embed(grads).permute(2, 3, 0, 1)
        return grads    

import math
class L2MultiheadAttention(nn.Module):
  

    def __init__(self, embed_dim, num_heads):
        super(L2MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.v_weight = nn.Parameter(torch.empty(embed_dim, num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_weight.view(self.embed_dim, self.embed_dim))
        nn.init.xavier_uniform_(self.v_weight.view(self.embed_dim, self.embed_dim))

    def forward(self, x, attn_mask=None, rm_nonself_grads=False, attn_multiplier=None):
       

        T, N, _ = x.shape

        q = k = torch.einsum("tbm,mhd->tbhd", x, self.q_weight)
        squared_dist = (torch.einsum('tbhd,tbhd->tbh', q, q).unsqueeze(1)
                        + torch.einsum('sbhd,sbhd->sbh', k, k).unsqueeze(0)
                        - 2 * torch.einsum('tbhd,sbhd->tsbh', q, k))
        attn_logits = -squared_dist / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask[..., None, None]
            attn_logits += attn_mask
        attn_weights = F.softmax(attn_logits, dim=1)  
        attn_weights = update_attn_weights(attn_weights, attn_multiplier)
        A = torch.einsum("mhd,nhd->hmn", self.q_weight, self.q_weight) / math.sqrt(self.head_dim)
        XA = torch.einsum("tbm,hmn->tbhn", x, A)
        PXA = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA)

        if rm_nonself_grads:
            
            q_detach = q.detach()
            k_detach = k.detach()
            attn_logits_keyonly = -(torch.einsum('tbhd,tbhd->tbh', q_detach, q_detach).unsqueeze(1)
                                    + torch.einsum('sbhd,sbhd->sbh', k, k).unsqueeze(0)
                                    - 2 * torch.einsum('tbhd,sbhd->tsbh', q_detach, k)) / math.sqrt(self.head_dim)
            attn_logits_queryonly = -(torch.einsum('tbhd,tbhd->tbh', q, q).unsqueeze(1)
                                      + torch.einsum('sbhd,sbhd->sbh', k_detach, k_detach).unsqueeze(0)
                                      - 2 * torch.einsum('tbhd,sbhd->tsbh', q, k_detach)) / math.sqrt(self.head_dim)

            attn_logits_keyonly = SelfonlyGradients.apply(attn_logits_keyonly)
            attn_logits = attn_logits_queryonly + (attn_logits_keyonly - attn_logits_keyonly.detach())
            if attn_mask is not None:
                attn_logits += attn_mask
            attn_weights = F.softmax(attn_logits, dim=1)
            attn_weights = update_attn_weights(attn_weights, attn_multiplier)

            
            selfonly_mask = ~(torch.triu(torch.ones(T, T), diagonal=1) + torch.tril(torch.ones(T, T), diagonal=-1)).bool()
            selfonly_attn_weights = attn_weights * selfonly_mask[..., None, None].to(attn_weights.device)
            
            PXA_vpath = torch.einsum("tsbh,sbhm->tbhm", selfonly_attn_weights.detach(), XA)
            PXA_spath = torch.einsum("tsbh,sbhm->tbhm", attn_weights, XA.detach())

            modified_PXA = PXA_spath + (PXA_vpath - PXA_vpath.detach())
            PXA = PXA.detach() + (modified_PXA - modified_PXA.detach())

        PXAV = torch.einsum("tbhm,mhd->tbhd", PXA, self.v_weight).reshape(T, N, self.embed_dim)
        return self.out_proj(PXAV), attn_weights.detach()       