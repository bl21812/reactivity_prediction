import math
import torch

import torch.nn as nn

class WatsonCrickAttentionLayer(torch.nn.Module):

    def __init__(self, size, score_matrix):
        super(WatsonCrickAttentionLayer, self).__init__()
        self.size = size
        self.score_matrix = torch.nn.Parameter(score_matrix, requires_grad=False)

        # Linear Components
        self.q_linear = torch.nn.Linear(size, size)
        self.k_linear = torch.nn.Linear(size, size)
        self.v_linear = torch.nn.Linear(size, size)

    def forward(self, x):
        # X = (batch_size, self.size)

        # Apply linear components
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Calculate attention
        score_attention = torch.matmul(q, k.transpose(-2, -1))
        weights_attention = torch.nn.functional.softmax(score_attention, dim=-1)

        # Output block
        return torch.matmul(weights_attention, v)


def wc_attention(query, key, value, wc_matrix, mask=None, dropout=None, softmax_wc=True):
    """ Same as 'Scaled Dot Product Attention', except we
        multiply the KV matrix by our watson crick matrix

    query, key, value (Tensor): (bs, h, N, d_k)
    wc_matrix (Tensor): (bs, N, N)

    """
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # wc_matrix: (bs, N, N) -> (bs, 1, N, N)
    if softmax_wc:
        scores = scores * wc_matrix.unsqueeze(1).softmax(dim=-1)
    else:
        scores = scores * wc_matrix.unsqueeze(1)

    p_attn = scores.softmax(dim=-1) # (bs, h, N, N)

    if dropout is not None:
        p_attn = dropout(p_attn)

    a = torch.matmul(p_attn, value) # (bs, h, N, d_k)
    return a, p_attn


class WatsonCrickMultiHeadedAttention(nn.Module):
    batch_first = True
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(WatsonCrickMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, wc_matrix, mask=None):
        """ Perform Multi-Head Attention with the Watson Crick Matrix

        Args:
            query, key, value (Tensor): (bs, N, d_model)
            wc_matrix (Tensor): (bs, N, N)
            mask (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: (bs, N, d_model)
        """
        # src shape: (bs, N, d_model)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = src.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        def head(lin, src):
            src = lin(src).view(nbatches, -1, self.h, self.d_k) # (bs, N, h, d_k)
            src = src.transpose(1, 2) # (bs, h, N, d_k)
            return src
        
        query, key, value = [
            head(lin, src)
            for lin, src in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = wc_attention(
            query, key, value, wc_matrix, mask=mask, dropout=self.dropout
        ) # (bs, h, N, d_k), (bs, h, N, N)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        ) # (bs, N, d_model)
        del src
        del key
        del value
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class WatsonCrickEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, h, d_ff, dropout):
        super(WatsonCrickEncoderLayer, self).__init__()
        self.self_attn = WatsonCrickMultiHeadedAttention(h, d_model, dropout)
        self.ff = nn.ModuleList([
            nn.Linear(d_model, d_ff),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        ])

        self.sublayer = [SublayerConnection(d_model, dropout) for _ in range(2)]

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.ff)


def build_wc_encoder(num_layers, num_frozen_layers, layer_cfg):
    encoder_layer= WatsonCrickEncoderLayer(
        d_model=layer_cfg['d_model'],
        h=layer_cfg['nhead'],
        d_ff=layer_cfg['dim_feedforward'],
        dropout=layer_cfg['dropout']
    )
    return nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
