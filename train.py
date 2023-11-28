import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import RNAInputDataset, BPPInputDataset
from models import Encoder, WatsonCrickAttentionLayer
from dataset import RNAInputDataset
from utils import load_df_with_secondary_struct

# ----- LOAD CONFIG -----
cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)

device = cfg['device']
pretrain = cfg['pretrain']
seq_length = cfg['data']['seq_length']
val_prop = cfg['data']['val_prop']
batch_size = cfg['data']['batch_size']

# ----- BUILD MODEL -----
model_type = cfg['model']['name'].lower()
weights = None if pretrain else cfg['model']['weights']
model_cfg = cfg['model'][model_type]
embedding_cfg = cfg['model']['embedding_cfg']
print(f"Building Model ({model_type})...")
if model_type == 'encoder':
    model = Encoder(
        embedding_cfg=embedding_cfg,
        num_layers=model_cfg['num_layers'],
        layer_cfg=model_cfg['layer_cfg'],
        finetune_cfg=model_cfg['finetune_cfg'],
        seq_length=seq_length,
        weights=weights,
    )
elif model_type == "attention":
    pass

# ----- LOAD + TRAIN LOOP -----
epochs = cfg['training']['epochs']
lr = cfg['training']['lr']

train_loss = []
val_loss = []
for epoch in range(epochs):

    df = pd.read_csv(cfg['data']['paths']['df'])

    if pretrain:
        print("Loading Secondary Structure for pre-training...")
        secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
        df, secondary_type = load_df_with_secondary_struct(df, secondary_struct_df)
        print(f"Loaded {df.shape[0]} seq with {secondary_type} secondary structure")

    # train/test splits
    df_train, df_val = train_test_split(df, test_size=val_prop)

    # data loaders
    print("Loading RNA+Secondary Datasets..." if pretrain else "Loading RNA+Reactivity Datasets...")
    ds_train = RNAInputDataset(df_train, pretrain=pretrain, seq_length=seq_length, device=device)
    ds_val = RNAInputDataset(df_val, pretrain=pretrain, seq_length=seq_length, device=device)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

# ----- TRAIN -----


# ----- SAVE MODEL -----
