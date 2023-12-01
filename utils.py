import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm


def load_df_with_secondary_struct(df, secondary_df, sample_size=46):
    """
    Adds secondary structure to training df

    df: training dataframe, 'sequence' column used for matching

    secondary_struct_df: concatenated secondary structure datatable
    (GPN15k, PK50, PK90, R1 combined) with columns for each secondary
    structure origin.

    All secondary packages are ultimately used (if sample_size = 46).
    The labels are pivoted into a single secondary_struct column. 
    Empty secondary structures are dropped and the indices are reset.

    Returns inner-merged df with a random secondary structure type
    """

    # Hardcoding the list of eligible secondary soures
    secondary_types = [
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
    ]

    # ability to select only a selection of the secondary types, default is entire list
    secondary_types_sample = random.sample(secondary_types, sample_size)
    secondary_types_sample = [col for col in secondary_types if col in secondary_df.columns]

    # Unpivot the secondary frame
    secondary_df = pd.melt(df, id_vars=['sequence'], value_vars=secondary_types_sample, value_name='secondary_struct')
    secondary_df = secondary_df.dropna(subset=['secondary_struct'])

    # Merge the secondary column
    df = pd.merge(df, secondary_df, how='left', left_on='sequence', right_on='sequence')

    # Drop empty secondary structures
    df = df.dropna(subset=['secondary_struct']).reset_index()

    # Drop sequences that do not match in length
    df = df[df['sequence'].str.len() == df['secondary_struct'].str.len()]
    df.reset_index(inplace=True)

    return df

def masked_mse(outputs, targets, mask):
    """
    computes mse loss.
    @return: loss (tensor)
    """

    print(outputs)
    print(targets)
    exit()
    
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

    if outputs.dim() > mask.dim():
        mask = mask.unsqueeze(-1).repeat(1, 1, outputs.size()[-1])
    
    mask = ~mask
    outputs = torch.masked_select(outputs, mask).float()
    targets = torch.masked_select(targets, mask).float()
    loss = F.cross_entropy(outputs, targets)

    return loss


def train(model, data_loader, loss_fn, optimizer, device):
    
    model = model.to(device)
    model.train()  # Set model in training mode
    
    total_loss = 0
    for (inputs, targets, mask) in tqdm(data_loader):
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
    for (inputs, targets, mask) in tqdm(data_loader):
        with torch.no_grad():
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            outputs = model(inputs, mask)
            loss = loss_fn(outputs, targets, mask)
    
            total_loss += loss.item()

    final_loss = total_loss / len(data_loader)
    return final_loss
