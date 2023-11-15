import torch
import pandas as pd

from torch.utils.data import Dataset

# Dataset that uses only RNA sequence as input
class RNAInputDataset(Dataset):

    def __init__(self, data_csv, pretrain=False, seq_length=512, device=None):
        '''
        data_csv should contain secondary structure for pretrain!
        '''

        self.device = device if device else 'cpu'

        self.df = pd.read_csv(data_csv)
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
