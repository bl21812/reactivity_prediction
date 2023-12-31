import torch

from embedding import SinusoidalPositionalEmbedding

class Encoder(torch.nn.Module):

    def __init__(self, num_layers, layer_cfg, output_cfg, finetune_cfg, embedding_cfg, seq_length, weights=None):

        # Load pretrained
        if weights:
            self.embedding = None
            self.model = torch.load(weights)
            # TODO: take off classification head
            # TODO: Freezing ?? (from finetune_cfg)

        else:

            self.embedding = torch.nn.Embedding(**embedding_cfg)
            self.position_embedding = SinusoidalPositionalEmbedding(
                num_positions=seq_length, 
                embedding_dim=embedding_cfg['embedding_dim']
            )
            # LAYER NORM HERE ??
        
            encoder_layer = torch.nn.TransformerEncoderLayer(**layer_cfg)
            self.model = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.output_norm = torch.nn.LayerNorm(layer_cfg['d_model'])
        self.output = torch.nn.Linear(layer_cfg['d_model'], 1)

    def forward(self, x, pad_mask):

        # pretrained model will have embedding layers
        if not self.embedding:
            x = self.model(x, pad_mask=pad_mask)

        else:

            # embedding
            embeddings = self.embedding(x)
            position_embeddings = self.position_embedding(x.shape)
            x = embeddings + position_embeddings

            # transformer layers
            x = self.model(x, src_key_padding_mask=pad_mask)

        # output block
        x = self.output_norm(x)
        x = self.output(x)
        return torch.squeeze(x)
    