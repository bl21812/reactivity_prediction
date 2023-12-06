import torch
import numpy as np
from pathlib import Path
import pandas as pd
import os
import tqdm
from torch.utils.data import Dataset
import time


# Dataset that uses only RNA sequence as input
class RNAInputDataset(Dataset):

    def __init__(self, df, pretrain=False, seq_length=512, device=None, watson_crick=False, wc_dir=None, do_wc=False):
        '''
        df should contain secondary structure for pretrain!
        '''

        self.device = device if device else 'cpu'
        self.df = df
        self.pretrain = pretrain
        self.seq_length = seq_length

        self.rna_encode = {
            'A': 1,
            'C': 2,
            'U': 3,
            'G': 4
        }

        self.secondary_struct_encode = {
            '(': 1,
            '[': 2,
            '{': 3,
            '<': 4,
            '.': 5,
            ')': 6,
            ']': 7,
            '}': 8,
            '>': 9
        }
        if watson_crick:
            assert wc_dir, "Must provide"
        self.do_wc = do_wc
        self.watson_crick = watson_crick
        self.wc_dir = Path(wc_dir)

        # put reactivities into one column if using them as labels
        if not pretrain:
            sub_df = self.df.filter(like='reactivity')
            sub_df.fillna(value=0., inplace=True)  # since some positions have no data
            reactivity_col = sub_df.apply(
                lambda row: row.to_list(),
                axis=1
            )
            self.df['reactivity'] = reactivity_col

    def __len__(self):
        return len(self.df)

    def get_corresponding_wc_matrix(self, sequence_id: str):
        if not self.do_wc:
            return []
        # print("BAD!")
        wc_matrix = np.load(f"/pub5/howard/reactivity_prediction/data/ribo_bpp_np/{sequence_id}.npy")
        x = torch.tensor(wc_matrix).to(self.device) if self.device != "cpu" else torch.tensor(wc_matrix)
        return x

    def __getitem__(self, idx):

        # load, one-hot encode, and pad rna sequence
        inp = self.df['sequence'].iloc[idx]
        inp = [self.rna_encode[c] for c in inp]
        pad_amount = self.seq_length - len(inp)
        inp += [0 for _ in range(pad_amount)]

        # padding mask
        pad_mask = [i == 0 for i in inp]

        # load, one-hot encode secondary structure
        if self.pretrain:
            label = self.df['secondary_struct'].iloc[idx]
            label = [
                self.secondary_struct_encode[c] if (c in self.secondary_struct_encode.keys()) else -1
                for c in label
            ]

            label += [0 for _ in range(pad_amount)]
            one_hot_label = []

            for i, idx in enumerate(label):
                temp = np.zeros(9)
                if idx > 0:
                    temp[idx - 1] = 1
                one_hot_label.append(temp)

            label = np.array(one_hot_label)

        # load, reactivities
        else:
            label = self.df['reactivity'].iloc[idx]
            label += [0 for _ in range(len(inp) - len(label))]

        # convert to tensor
        inp = torch.tensor(inp, dtype=torch.long)  # LongTensor for Embedding layer
        label = torch.tensor(label)

        if self.watson_crick:
            label = label.unsqueeze(-1)

        pad_mask = torch.tensor(pad_mask)

        # send to device
        if not (self.device == 'cpu'):
            inp = inp.to(self.device)
            label = label.to(self.device)

        if self.watson_crick:
            return inp, label, pad_mask.unsqueeze(1).to(self.device), self.get_corresponding_wc_matrix(self.df['sequence_id'].iloc[idx])
        else:
            return inp, label, pad_mask, None
