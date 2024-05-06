import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence

# Constants
MAX_LEN_ANTIGEN = 11
MAX_LEN_TCR = 21
EMBEDDING_DIM = 256
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'

char_to_index = {ch: idx for idx, ch in enumerate(AMINO_ACIDS)}

# Helper Functions
def encode_sequence(seq, max_len):
    encoded = torch.zeros(max_len, dtype=torch.long)
    for i, char in enumerate(seq[:max_len]):  
        encoded[i] = char_to_index.get(char, char_to_index['X'])
    return encoded 


def load_data(filename):
    data = pd.read_csv(filename)
    # Remove leading and trailing whitespaces from column names
    data.columns = data.columns.str.strip()
    print("Columns:", data.columns)  # Print column names to verify
    antigens = data['antigen']
    tcrs = data['TCR']
    interactions = data['interaction']
    antigen_seqs = [encode_sequence(seq, MAX_LEN_ANTIGEN) for seq in antigens]
    tcr_seqs = [encode_sequence(seq, MAX_LEN_TCR) for seq in tcrs]
    return TensorDataset(torch.stack(antigen_seqs), torch.stack(tcr_seqs), torch.tensor(interactions))


# Define the dataset class
class InteractionDataset(Dataset):
    def __init__(self, antigens, tcrs, interactions):
        self.antigens = torch.stack(antigens)
        self.tcrs = torch.stack(tcrs)
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.antigens[idx], self.tcrs[idx], self.interactions[idx]

   
    
if __name__ == '__main__':
    dataset = load_data('datatest.csv')
    for i in range(10):
        print(dataset[i])
    