from torch.utils.data import DataLoader, Dataset
import pandas as pd


class ATCDataset(Dataset):
    """
    Dataset class for ATC data
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path (str): path to the data
        """
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
