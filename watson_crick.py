import math
import torch
import torch.nn as nn


def wc_attention(query, key, value, wc_matrix, mask=None, dropout=None, softmax_wc=True, residual=True):
    """ Same as 'Scaled Dot Product Attention', except we
        multiply the KV matrix by our watson crick matrix

    query, key, value (Tensor): (bs, h, N, d_k)
    wc_matrix (Tensor): (bs, N, N)

    """
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    if softmax_wc:
        scores = scores * wc_matrix.unsqueeze(1).softmax(dim=-1) + (scores if residual else 0)
    else:
        scores = scores * wc_matrix.unsqueeze(1) + (scores if residual else 0)

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
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
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
        del query
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


class WatsonCrickEncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, h, d_ff, dropout):
        super(WatsonCrickEncoderLayer, self).__init__()
        self.self_attn = WatsonCrickMultiHeadedAttention(h, d_model, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.sublayer = [SublayerConnection(d_model, dropout) for _ in range(2)]

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.ff)


def build_wc_encoder(num_layers, num_frozen_layers, layer_cfg):
    encoder_layer = WatsonCrickEncoderLayer(
        d_model=layer_cfg['d_model'],
        h=layer_cfg['nhead'],
        d_ff=layer_cfg['dim_feedforward'],
        dropout=layer_cfg['dropout']
    )
    return nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
