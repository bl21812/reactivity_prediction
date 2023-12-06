import os
import pandas as pd
import time
import torch
import torch.optim as optim
import utils
import yaml

import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import RNAInputDataset
from models import Encoder
from watson_crick import WatsonCrickEncoder
from utils import load_df_with_secondary_struct

# ----- LOAD CONFIG -----

cfg = yaml.load(open('config.yml', 'r'), Loader=yaml.CLoader)

device = cfg['device']
pretrain = cfg['pretrain']
experiment = cfg['data']['experiment']
seq_length = cfg['data']['seq_length']
val_prop = cfg['data']['val_prop']
batch_size = cfg['data']['batch_size']

epochs = cfg['training']['epochs']
lr = float(cfg['training']['lr'])
save = cfg['model']['save']

# ----- BUILD MODEL -----

model_type = cfg['model']['name'].lower()
weights = None if pretrain else cfg['model']['weights']
model_cfg = cfg['model'][model_type]
embedding_cfg = cfg['model']['embedding_cfg']
print(f"Building Model ({model_type})...")

if model_type == 'encoder':
    encoder_type = Encoder
elif model_type == 'watson_crick':
    encoder_type = WatsonCrickEncoder
else:
    raise NotImplementedError()

model = encoder_type(
    embedding_cfg=embedding_cfg,
    num_layers=model_cfg['num_layers'],
    layer_cfg=model_cfg['layer_cfg'],
    seq_length=seq_length,
    weights=weights,
)

# ----- LOAD + TRAIN LOOP -----

loss_fn = utils.masked_cross_entropy if pretrain else utils.masked_mse
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loss = []
val_loss = []

print("Loading training dataframe...")
df_raw = pd.read_csv(cfg['data']['paths']['df'])
df_exp = df_raw[df_raw['experiment_type'] == experiment]

if pretrain:
    print("Loading Secondary Structure for pre-training...")
    secondary_struct_df = pd.read_csv(cfg['data']['paths']['secondary_struct_df'])
    df = load_df_with_secondary_struct(df_exp, secondary_struct_df)  # optional = sample_size
    print(f"Loaded {df.shape[0]} sequences with secondary structure")
else:
    df = df_exp

df_train, df_val = train_test_split(df, test_size=val_prop)

print("Loading RNA+Secondary Datasets..." if pretrain else "Loading RNA+Reactivity Datasets...")
ds_train = RNAInputDataset(
    df_train, pretrain=pretrain,
    seq_length=seq_length, device=device,
    watson_crick=model_type == 'watson_crick',
    wc_dir=cfg['data']['paths']['bpp_files'],
    do_wc=cfg['data'].get('do_wc', True)
)
ds_val = RNAInputDataset(
    df_val, pretrain=pretrain,
    seq_length=seq_length, device=device,
    watson_crick=model_type == 'watson_crick',
    wc_dir=cfg['data']['paths']['bpp_files'],
    do_wc=cfg['data'].get('do_wc', True)
)
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

print("Training model...")
best_val_loss = 1000000
best_model = None
for epoch in range(epochs):    
    avg_train_loss = utils.train(
        model=model, 
        data_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )

    avg_val_loss = utils.test(
        model=model,
        data_loader=val_loader,
        loss_fn=loss_fn,
        device=device
    )

    train_loss.append(avg_train_loss)
    val_loss.append(avg_val_loss)

    with open("epoch.csv", "a+") as file:
        file.write(f"{avg_train_loss},{avg_val_loss}\n")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = deepcopy(model)

    print(f"Epoch {epoch + 1}:".ljust(16), f"Train Loss: {avg_train_loss:.4f}".rjust(20), f"Validation Loss: {avg_val_loss:.4f}".rjust(20))

# ----- SAVE MODEL -----

if save:

    save = os.path.join(save, time.strftime("%Y%m%d_%H%M%S"))

    if not os.path.exists(save):
        os.makedirs(save)

    print(f'Saving to directory: {save}')

    # save model
    filename = 'model.pt'
    if best_model:
        torch.save(best_model, f=os.path.join(save, filename))
    else:
        torch.save(model, f=os.path.join(save, filename))

    # save plots
    xs = [i+1 for i in range(epochs)]
    plt.plot(xs, train_loss, color='b', label='train')
    plt.plot(xs, val_loss, color='r', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save, 'loss.png'))

    # save plots separately
    plt.clf()
    plt.plot(xs, train_loss, color='b', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save, 'train_loss.png'))

    plt.clf()
    plt.plot(xs, val_loss, color='r', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save, 'val_loss.png'))
