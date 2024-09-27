from torch.utils.data import Dataset
import pandas as pd
import numpy as np  
import torch

def create_vector(group, target_length=5000):
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
    
class ATCDataset(Dataset):
    """
    Dataset class for ATC data
    """
        
    def __init__(self, in_data_path, out_data_path):
        """
        Args:
            data_path (str): path to the data in csv format
        """
        self.df = pd.read_csv(
            in_data_path,
            delimiter=",",
            header=0
            )
        self.out_df = pd.read_csv(
            out_data_path,
            delimiter=",",
            header=0
            )
        
        # build output data for the loss function

        # labels for vector of the ouput boxes 
        labels = [
            "OM0000", "OM0001", "OM0002", "OM0003", "OM0004", "OM0005", "OM0006", "OM0007", 
            "OM0008", "OM0009", "OM0010", "PAL01", "PAL02", "PAL03", "PAL04", "PAL05", "PAL06", 
            "PAL07"
        ]

        # result vector contains number of occurences of each box for each group of items
        result_vector = {}
        for group in self.out_df['GroupDelivery'].unique():
            result_vector[group] = np.zeros((labels.__len__()))
            
        for _, row in self.out_df.iterrows():
            cartonName = row['UsedCarton'].strip().upper()
            result_vector[row['GroupDelivery']][labels.index(cartonName)] += 1
        self.result_vector = result_vector
        
        # now prepare the input data
        # Apply the function to each GroupDelivery
        grouped = self.df.groupby('GroupDelivery').apply(create_vector, include_groups=False).reset_index()

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
        gd = self.groupDelivery.iloc[idx]
        # Ensure the vectors are returned as numpy arrays
        input_vector = torch.tensor(self.vectors.iloc[idx], dtype=torch.float32)  # Convert input vector to tensor
        output_vector = torch.tensor(self.result_vector[gd], dtype=torch.float32)  # Convert result vector to tensor
        return gd, input_vector, output_vector

def create_feature_vector(input):
    vector = input['Vector']
    
    
class ATCDataset_v2(Dataset):
    
    def __init__(self, in_data_path, out_data_path):
        """
        Args:
            data_path (str): path to the data in csv format
        """
        self.df = pd.read_csv(
            in_data_path,
            delimiter=",",
            header=0
            )
        self.out_df = pd.read_csv(
            out_data_path,
            delimiter=",",
            header=0
            )
        
        # build OUTPUT DATA for the loss function
        # ========================================
        # labels for vector of the ouput boxes 
        labels = [
            "OM0000", "OM0001", "OM0002", "OM0003", "OM0004", "OM0005", "OM0006", "OM0007", 
            "OM0008", "OM0009", "OM0010", "PAL01", "PAL02", "PAL03", "PAL04", "PAL05", "PAL06", 
            "PAL07"
        ]

        # result vector contains number of occurences of each box for each group of items
        result_vector = {}
        for group in self.out_df['GroupDelivery'].unique():
            result_vector[group] = np.zeros((labels.__len__()))
            
        for _, row in self.out_df.iterrows():
            cartonName = row['UsedCarton'].strip().upper()
            result_vector[row['GroupDelivery']][labels.index(cartonName)] += 1
        self.result_vector = result_vector
        
        # build INPUT DATA 
        # ========================================
        
        # Apply the function to each GroupDelivery
        def vectorize(group):
            return group[['X', 'Y', 'Z', 'Weight', 'Qty']].values.flatten().tolist()
        
        grouped = self.df.groupby('GroupDelivery').apply(group, include_groups=False).reset_index()

        # Rename columns
        grouped.columns = ['GroupDelivery', 'Vector']
        
        # store vectors and group delivery
        self.vectors = grouped.Vector
        self.groupDelivery = grouped.GroupDelivery