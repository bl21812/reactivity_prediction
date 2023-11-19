import yaml
import pandas as pd

from models import Encoder
from utils import load_df_with_secondary_struct

cfg = yaml.load('config.yml')

# ----- LOAD DATA -----

pretrain = cfg['pretrain']
seq_length = cfg['data']['seq_length']

df = pd.read_csv(cfg['data']['paths']['df'])

if pretrain:
    secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
    df = load_df_with_secondary_struct(df, secondary_struct_df)

# train/test splits
# data loaders

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
