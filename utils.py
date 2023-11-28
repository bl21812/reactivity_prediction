import os
import pandas as pd
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

def load_df_with_secondary_struct(df, secondary_struct_df):
    """
    Adds secondary structure to training df

    df: training dataframe, 'sequence' column used for matching

    secondary_struct_df: concatenated secondary structure datatable
    (GPN15k, PK50, PK90, R1 combined) with columns for each secondary
    structure origin.

    secondary_type is randomly sampled from a hard-coded list of 
    all secondary structures. Note that there are varying amount of
    empty cells per secondary-structure origin.

    Returns inner-merged df with a random secondary structure type
    """
    
    # Hardcoding the list of eligible secondary soures
    secondary_type = np.random.choice([
        'eterna_nupack',
        'eterna_eternafold+threshknot',
        'vienna2_mfe', 
        'contrafold2_mfe',
        'eternafold_mfe', 
        'e2efold_mfe', 
        'hotknots_mfe', 
        'ipknots_mfe',
        'knotty_mfe', 
        'pknots_mfe', 
        'spotrna_mfe', 
        'vienna[threshknot]_mfe',
        'vienna[hungarian]_mfe', 
        'eternafold[threshknot]_mfe',
        'eternafold[hungarian]_mfe', 
        'contrafold[threshknot]_mfe',
        'contrafold[hungarian]_mfe', 
        'nupack[threshknot]_mfe',
        'nupack[hungarian]_mfe', 
        'shapify_mfe', 
        'eternafold+hfold_1',
        'eternafold+hfold_2', 
        'eternafold+hfold_3', 
        'eternafold+hfold_4',
        'eternafold+hfold_final', 
        'nupack_mfe-pk', 
        'nupack-pk.threshknot',
        'nupack-pk.hungarian',
        'nupack.threshknot',
        'nupack.hungarian', 
        'hotknots', 
        'ipknots', 
        'knotty', 
        'spotrna', 
        'nupack_pk', 
        'vienna_2[threshknot]',
        'vienna_2[hungarian]', 
        'eternafold[threshknot]',
        'eternafold[hungarian]',
        'contrafold_2[threshknot]',
        'contrafold_2[hungarian]',
        'nupack[threshknot]',
        'nupack[hungarian]',
        'nupack-pk[threshknot]',
        'nupack-pk[hungarian]',
        'shapify-hfold', 
    ])

    secondary_struct = []
    for idx, row in df.iterrows():
        seq = row['sequence']
        sub_df = secondary_struct_df.loc[secondary_struct_df['sequence'] == seq]
        if len(sub_df) > 0:
            secondary_struct.append(sub_df.iloc[0][secondary_type])
        else:
            secondary_struct.append(None)
    df['secondary_struct'] = secondary_struct

    return df.dropna(subset=('secondary_struct')), secondary_type


def masked_mse(outputs, targets, mask):
    """
    computes mse loss.
    @return: loss (tensor)
    """
    mask = ~mask
    outputs = torch.masked_select(outputs, mask).float()
    targets = torch.masked_select(targets, mask).float()
    loss = F.mse_loss(outputs, targets)

    return loss


def masked_cross_entropy(outputs, targets, mask):
    """
    computes cross entropy loss.
    @return: loss (tensor)
    """
    mask = ~mask
    outputs = torch.masked_select(outputs, mask).float()
    targets = torch.masked_select(targets, mask).float()
    loss = F.cross_entropy(outputs, targets)

    return loss


def train(model, data_loader, loss_fn, optimizer, device):
    
    model = model.to(device)
    model.train()  # Set model in training mode
    
    total_loss = 0
    for i, (inputs, targets, mask) in enumerate(data_loader):
        optimizer.zero_grad()
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs, mask)
        loss = loss_fn(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    final_loss = total_loss / len(data_loader)
    return final_loss

def test(model, data_loader, loss_fn, device):
    
    model = model.to(device)

    total_loss = 0
    for i, (inputs, targets, mask) in enumerate(data_loader):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs, mask)
        loss = loss_fn(outputs, targets, mask)

        total_loss += loss.item()

    final_loss = total_loss / len(data_loader)
    return final_loss