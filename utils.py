import os
import pandas as pd
import numpy as np

def load_df_with_secondary_struct(df, secondary_struct_df):
    
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
    ###
    return loss

def train(model, train_loader, loss_fn, optimizer, epoch):
    """
    trains a model for one epoch (one pass through the entire training data).
    @return: final loss (float)
    """
    total_loss = 0
    loss_history = []

    model = model.to(device)
    model.train()  # Set model in training mode
    for i, (inputs, targets, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        outputs = model(inputs, mask)
        loss = masked_mse(outputs, targets, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    final_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}")
    return final_loss
