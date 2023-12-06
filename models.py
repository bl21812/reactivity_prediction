import math
import torch
import torch.nn as nn

from embedding import SinusoidalPositionalEmbedding


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, layer_cfg, embedding_cfg, seq_length, weights=None, num_frozen_layers=0):
        super().__init__()

        if weights:
            # Load pretrained model
            pt_model = torch.load(weights)
            self.embedding = None #pt_model.embedding
            self.position_embedding = pt_model.position_embedding
            self.input_layer_norm = pt_model.input_layer_norm
            self.model = pt_model.model
            self.output_norm = pt_model.output_norm

            # Create a new output layer
            self.output = torch.nn.Linear(layer_cfg['d_model'], 1)

            # Freeze the first 'num_frozen_layers'
            for i, param in enumerate(self.model.parameters()):
                if i < num_frozen_layers:
                    param.requires_grad = False


        else:
            self.embedding = torch.nn.Embedding(**embedding_cfg)
            self.position_embedding = SinusoidalPositionalEmbedding(
                num_positions=seq_length,
                embedding_dim=embedding_cfg['embedding_dim']
            )
            self.input_layer_norm = torch.nn.LayerNorm(layer_cfg['d_model'])

            encoder_layer = self.encoder_layer_type(**layer_cfg)
            self.model = self.encoder_type(encoder_layer=encoder_layer, num_layers=num_layers)

        self.output_norm = torch.nn.LayerNorm(layer_cfg['d_model'])

        # Labels are one-hot encodings for pretrain, but floats for finetune
        output_dim = 1
        self.output = torch.nn.Linear(layer_cfg['d_model'], output_dim)

    @property
    def encoder_layer_type(self):
        return torch.nn.TransformerEncoderLayer

    @property
    def encoder_type(self):
        return torch.nn.TransformerEncoder

    def forward(self, x, pad_mask):

        # pretrained model will have embedding layers
        if not self.embedding:
            x = self.model(x, pad_mask=pad_mask)

        else:
            # embedding
            embeddings = self.embedding(x)
            position_embeddings = self.position_embedding(x.shape)
            x = embeddings + position_embeddings
            x = self.input_layer_norm(x)

            # transformer layers
            x = self.model(x, src_key_padding_mask=pad_mask)

        # output block
        x = self.output_norm(x)
        x = self.output(x)
        return x.unsqueeze(-1)
