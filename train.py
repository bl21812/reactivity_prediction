import yaml

from models import Encoder

cfg = yaml.load('config.yml')

# ----- LOAD DATA -----
pretrain = cfg['pretrain']
seq_length = cfg['data']['seq_length']

# TODO: DETERMINE max seq length in train & test and set accordingly in cfg

# ----- BUILD MODEL -----
model_type = cfg['model']['name'].lower()
weights = None if pretrain else cfg['model']['weights']
model_cfg = cfg['model'][model_type]
embedding_cfg = cfg['model']['embedding_cfg']
if model_type == 'encoder':
    model = Encoder(
        embedding_cfg=embedding_cfg,
        num_layers=model_cfg['num_layers'], 
        layer_cfg=model_cfg['layer_cfg'],
        seq_length=seq_length,
        weights=weights, 
    )

# ----- TRAIN -----
