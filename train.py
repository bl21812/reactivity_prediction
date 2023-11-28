import yaml
import pandas as pd


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import RNAInputDataset, BPPInputDataset
from models import Encoder, WatsonCrickAttentionLayer
from dataset import RNAInputDataset
from utils import load_df_with_secondary_struct
import utils

# ----- LOAD CONFIG -----
cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)

device = cfg['device']
pretrain = cfg['pretrain']
seq_length = cfg['data']['seq_length']
val_prop = cfg['data']['val_prop']
batch_size = cfg['data']['batch_size']

epochs = cfg['training']['epochs']
lr = cfg['training']['lr']

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
loss_fn = utils.masked_cross_entropy if pretrain else utils.masked_mse
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loss = []
val_loss = []

for epoch in range(epochs):

    df = pd.read_csv(cfg['data']['paths']['df'])

    if pretrain:
        print("Loading Secondary Structure for pre-training...")
        secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
        df, secondary_type = load_df_with_secondary_struct(df, secondary_struct_df)
        print(f"Loaded {df.shape[0]} seq with {secondary_type} secondary structure")

    df_train, df_val = train_test_split(df, test_size=val_prop)

    print("Loading RNA+Secondary Datasets..." if pretrain else "Loading RNA+Reactivity Datasets...")
    ds_train = RNAInputDataset(df_train, pretrain=pretrain, seq_length=seq_length, device=device)
    ds_val = RNAInputDataset(df_val, pretrain=pretrain, seq_length=seq_length, device=device)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

    print("Training model...")
    avg_train_loss = utils.train(
        model=model, 
        data_loader=ds_train,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    avg_val_loss = utils.test(
        model=model, 
        data_loader=ds_train,
        loss_fn=loss_fn,
        device=device
    )

    train_loss.append(avg_train_loss)
    val_loss.append(avg_val_loss)

    print(f"Epoch {epoch + 1}:".ljust(16), f"Train Loss: {avg_train_loss:.4f}".rjust(20), f"Validation Loss: {avg_val_loss:.4f}.rjust(20)")
    
