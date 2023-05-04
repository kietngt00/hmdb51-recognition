import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):    
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = Dataset(train_path)
        self.val_dataset = Dataset(val_path)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)