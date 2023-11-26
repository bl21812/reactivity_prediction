import torch
import numpy as np
import pandas as pd
import os
import tqdm
from torch.utils.data import Dataset

# Dataset that uses only RNA sequence as input
class RNAInputDataset(Dataset):

    def __init__(self, df, pretrain=False, seq_length=512, device=None):
        '''
        df should contain secondary structure for pretrain!
        '''

        self.device = device if device else 'cpu'

        # if pretrain - remove rows without secondary structures
        if pretrain:
            df.dropna(subset=['secondary_struct'], inplace=True)

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
            ')': 2, 
            '.': 3
        }
        
        # put reactivities into one column if using them as labels
        if not pretrain:
            sub_df = self.df.filter(like='reactivity')
            reactivity_col = sub_df.apply(
                lambda row: row.to_list(), 
                axis=1
            )
            self.df['reactivity'] = reactivity_col

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        # load, one-hot encode, and pad rna sequence
        inp = self.df['sequence'][idx]
        inp = [self.rna_encode[c] for c in inp]
        pad_amount = self.seq_length - len(inp)
        inp += [0 for _ in range(pad_amount)]

        # padding mask
        pad_mask = inp == 0

        # load, one-hot encode secondary structure
        if self.pretrain:
            label = self.df['secondary_struct'][idx]
            label = [self.secondary_struct_encode[c] for c in label]

        # load, one-hot encode reactivities
        else:
            label = self.df['reactivity'][idx]

        # pad label
        label += [0 for _ in range(pad_amount)]

        # convert to tensor
        inp = torch.tensor(inp, dtype=torch.LongTensor)  # LongTensor for Embedding layer
        label = torch.tensor(label)

        # send to device
        if not (self.device == 'cpu'):
            inp = inp.to(self.device)
            label = label.to(self.device)

        return inp, label, pad_mask
    


class BPPInputDataset(Dataset):

    def __init__(self, df, bpp_dir, fb_range=range(16), seq_length=512, device=None):
        '''
        Load the 16x16x16x525=2.15M text files into dataframes and store in a dict

        df: subset of training data. Only used for id's and reactivity labels

        fb_range: option to limit first-byte range to load. The idea is for efficient 
        test/train split where we load the first 15/16 portions for train, and remaining 
        1/16 for val (for example). Default is to search all files for df's ids.

        seq_length: maximum sequence length for the entire dataset. Assuming we want to
        return a bpp matrix with zero-padding for consistent sizings.
 
        '''

        self.ids = df.sequence_id
        self.bpp_dir = bpp_dir
        self.fb_range = fb_range
        self.seq_length = seq_length
        self.device = device
        self.labels = df.filter(like='reactivity').apply(
            lambda row: row.to_list(), 
            axis=1
        )

        bpp_dict = {}
        for i in tqdm.tqdm(np.array(os.listdir(self.bpp_dir))[fb_range]):
            for j in os.listdir(os.path.join(self.bpp_dir, i)):
                for k in os.listdir(os.path.join(self.bpp_dir, i, j)):
                    for fn in os.listdir(os.path.join(self.bpp_dir, i, j, k)):
                        if (fn.split(sep='.')[0] in df.sequence_id):
                            path = os.path.join(self.bpp_dir, i, j, k, fn)
                            bpp = pd.read_csv(path, sep=' ', header=None, names=['b1', 'b2', 'p'])
                            bpp_dict[fn.split(sep='.')[0]] = bpp
        self.bpp_dict = bpp_dict


    def __len__(self):
        return len(self.bpp_dict.keys())
    

    def __getitem__(self, idx):
        '''
        Return a (seq_length,seq_length) matrix with base-pair probabilities for a given index

        Note that diagonals will be set to 1. This is a key assumption that can be revisited.
        '''

        id = self.ids[idx]
        bpp = self.bpp_dict[id]

        base1 = bpp.b1.to_numpy()
        base2 = bpp.b2.to_numpy()
        prob = bpp.p.to_numpy()

        bpp_matrix = np.identity(self.seq_length)
        bpp_matrix[base1-1, base2-1] = prob #convert from 1-indexed to 0-indexed system
        bpp_matrix = torch.tensor(bpp_matrix)

        label = self.labels[idx]
        label.resize(self.seq_length, refcheck=False) #zero-pad inplace

        # send to device (copied from RNAInputDataset above)
        if not (self.device == 'cpu'):
            bpp_matrix = bpp_matrix.to(self.device)
            label = label.to(self.device)

        return bpp_matrix, label