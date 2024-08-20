from torch.utils.data import DataLoader, Dataset
import pandas as pd


class ATCDataset(Dataset):
    """
    Dataset class for ATC data
    """
    
    def _create_vector(group, target_length=5000):
        """
        Helper function -Flatten the values into a single list and pad it to the target length
        Args:
            group (pandas dataframe): group of products
        """
        vector = group[['X', 'Y', 'Z', 'Weight', 'Qty']].values.flatten().tolist()
        
        # Pad with zeros or truncate to fit the target length
        if len(vector) < target_length:
            vector += [0] * (target_length - len(vector))
        else:
            vector = vector[:target_length]
        
        return vector
        
    def __init__(self, data_path):
        """
        Args:
            data_path (str): path to the data in csv format
        """
        self.df = pd.read_csv(
            data_path,
            delimiter=",",
            header=0
            )
        
        # Apply the function to each GroupDelivery
        grouped = self.df.groupby('GroupDelivery').apply(self._create_vector, include_groups=False).reset_index()

        # Rename columns
        grouped.columns = ['GroupDelivery', 'Vector']
        
        # store vectors and group delivery
        self.vectors = grouped.Vector
        self.groupDelivery = grouped.GroupDelivery

    def __len__(self):
        # the length of the self.Vectors (same as length of self.groupDelivery) is the length of the samples in the dataset
        # as sample we meant the products per order
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.groupDelivery.iloc[idx], self.vectors.iloc[idx]
