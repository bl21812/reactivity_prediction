import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import RNAInputDataset, BPPInputDataset
from models import Encoder, WatsonCrickAttentionLayer
from dataset import RNAInputDataset
from utils import load_df_with_secondary_struct

cfg = yaml.load('config.yml')

# ----- LOAD DATA -----

device = cfg['device']
pretrain = cfg['pretrain']
seq_length = cfg['data']['seq_length']
val_prop = cfg['data']['val_prop']
batch_size = cfg['data']['batch_size']

'''
UTILITY TO DETERMINE MAX SEQUENCE LENGTH
For our information - run once to set param accordingly in config

df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_sequences.csv')

max_train = df_train['sequence'].str.len().max()
max_test = df_test['sequence'].str.len().max()

max_seq_length = max(max_train, max_test)
print(f'Longest sequence: {max_seq_length}')
'''

df = pd.read_csv(cfg['data']['paths']['df'])

if pretrain:
    secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
    df = load_df_with_secondary_struct(df, secondary_struct_df)

# train/test splits
df_train, df_val = train_test_split(df, test_size=val_prop)

# data loaders
ds_train = RNAInputDataset(df_train, pretrain=pretrain, seq_length=seq_length, device=device)
ds_val = RNAInputDataset(df_val, pretrain=pretrain, seq_length=seq_length, device=device)
#bpp_train = BPPInputDataset(df_train, bpp_dir=cfg['data']['paths']['bpp_files'])
#bpp_val = BPPInputDataset(df_val, bpp_dir=cfg['data']['paths']['bpp_files'])

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

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
elif model_type == "attention":
    # TODO: Add score_matrix to Attention Layer
    model = WatsonCrickAttentionLayer(size=seq_length, score_matrix=None)


# ----- TRAIN -----
