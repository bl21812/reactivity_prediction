import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models import Encoder
from dataset import RNAInputDataset
from utils import load_df_with_secondary_struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

with open('config.yml', 'r') as file:
    cfg = yaml.safe_load(file)
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

train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=True)

# TODO: DETERMINE max seq length in train & test and set accordingly in cfg

# ----- BUILD MODEL -----
model_type = cfg['model']['name'].lower()
weights = None if pretrain else cfg['model']['weights']
model_cfg = cfg['model'][model_type]
embedding_cfg = cfg['model']['embedding_cfg']
output_cfg = '' # todo
finetune_cfg = '' # todo

if model_type == 'encoder':
    model = Encoder(
        embedding_cfg=embedding_cfg,
        num_layers=model_cfg['num_layers'], 
        layer_cfg=model_cfg['layer_cfg'],
        seq_length=seq_length,
        weights=weights, 
        output_cfg=output_cfg,
        finetune_cfg=finetune_cfg
    )

# ----- TRAIN -----
def masked_mse(outputs, targets, mask):
    """
    computes mse loss.
    @return: loss (tensor)
    """
    mask = ~mask
    outputs = torch.masked_select(outputs, mask).float()  
    targets = torch.masked_select(targets, mask).float()  
    loss = F.mse_loss(outputs, targets)
    ###
    return loss

def train(model, train_loader, loss_fn, optimizer, epoch):
    """
    trains a model for one epoch.
    @return: final loss (float)
    """
    total_loss = 0
    model = model.to(device)
    model.train()  # Set model in training mode
    for i, (inputs, targets, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs, mask)
        loss = loss_fn(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    final_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}")
    return final_loss

EPOCHS = 10
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr) 
loss_fn = masked_mse 
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, loss_fn, optimizer, epoch)
    #test_loss = test(model, train_loader, loss_fn, optimizer, epoch) # todo

    train_losses.append(train_loss)
    #test_losses.append(test_loss) # todo
