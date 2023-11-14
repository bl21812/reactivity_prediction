import torch

class Encoder(torch.nn.Module):

    def __init__(self, num_layers, layer_cfg, output_cfg, finetune_cfg, embedding_cfg, weights=None):

        self.embedding = torch.nn.Embedding(**embedding_cfg)
        # TODO: Positional embeddings?

        if weights:
            self.model = torch.load(weights)
            # TODO: Freezing ?? (from finetune_cfg)
        
        else:
            encoder_layer = torch.nn.TransformerEncoderLayer(**layer_cfg)
            self.model = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.output = None  # TODO: ADD OUTPUT LAYER !!!

    def forward(self, x):
        x = self.embedding(x)
        x = self.model(x)
        return self.output(x)
    

