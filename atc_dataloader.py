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

# dataset creates input data for the model as 1D vector of params per group of items (each item with its params)
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
        # Ensure the vectors are returned as torch arrays
        input_vector = torch.tensor(self.vectors.iloc[idx], dtype=torch.float32)  # Convert input vector to tensor
        output_vector = torch.tensor(self.result_vector[gd], dtype=torch.float32)  # Convert result vector to tensor
        return gd, input_vector, output_vector

# this is create feature vector
# it creates a single vector, that describes most the group, from the group of items
def create_feature_vector(group): 
    """
    Create a feature vector for a group of items
    Args:
        group (pandas dataframe): group of products
    Returns:
        list (list): feature vector
    """
    gr = group[['X','Y','Z','Weight']].copy()
    gr_all = group[['X','Y','Z','Weight','Qty']].copy()
    
    # get the sum of the values
    x_sum,y_sum,z_sum,w_sum = gr.sum(numeric_only=True, axis=0)
    # get the mean of the values
    x_mean, y_mean, z_mean, w_mean = gr.mean(numeric_only=True, axis=0)
    # get the standard deviation of the values
    x_std, y_std, z_std, w_std = gr.std(numeric_only=True, axis=0)
    # get the median of the values
    x_median, y_median, z_median, w_median = gr.median(numeric_only=True, axis=0)
    # get qty of items to boxes according to the weight
    bins_weight = [0, 1, 3, 6, 9, 13, 20, 25, 30, 100]  # Example weight ranges for bins
    labels_weight = ['0-1kg', '1-3kg', '3-6kg', '6-9kg','9-13kg','13-20kg','20-25kg','25-30kg','30-nkg']  # Labels for bins
    gr_all.loc[:,'Weight_bin'] = pd.cut(gr['Weight'], bins=bins_weight, labels=labels_weight, right=False)
    # get qty of items to boxes according to the volume
    gr.loc[:,'Volume'] = gr['X']*gr['Y']*gr['Z']
    
    bins_volume = [0,1000,2000,5000,10000,30000,50000,80000,100000,150000,250000,400000,600000,800000,1000000,10000000] # cm3
    labels_volume = [ 
        '0-1dm3','1-2dm3','2-5dm3','5-10dm3', '10-30dm3', '30-50dm3', '50-80dm3', '80-100dm3', 
        '100-150dm3', '150-250dm3', '250-400dm3', '400-600dm3', 
        '600-800dm3', '800-1000dm3', '1000-10000dm3'
    ]  # Labels for bins (written in dm3 for better readability)
    gr_all.loc[:,'Volume_bin'] = pd.cut(gr['Volume'], bins=bins_volume, labels=labels_volume, right=False)

    # return the values as a pandas series
    feature_vector = pd.Series(
        [
            x_sum, y_sum, z_sum, w_sum,
            x_mean, y_mean, z_mean, w_mean,
            x_std, y_std, z_std, w_std,
            x_median, y_median, z_median, w_median
        ], 
        index=[
            'X_sum', 'Y_sum', 'Z_sum', 'Weight_sum',
            'X_mean', 'Y_mean', 'Z_mean', 'Weight_mean',
            'X_std', 'Y_std', 'Z_std', 'Weight_std',
            'X_median', 'Y_median', 'Z_median', 'Weight_median'
        ])
    
    # Get the counts of each weight bin
    weight_bin_qty_sum = gr_all.groupby('Weight_bin',observed=True)['Qty'].sum()
    volume_bin_qty_sum = gr_all.groupby('Volume_bin',observed=True)['Qty'].sum()
    # Append the weight bin counts to the feature vector
    for bin_label in labels_weight:
        feature_vector[f'bin_{bin_label}_count'] = weight_bin_qty_sum.get(bin_label, 0)
    for bin_label in labels_volume:
        feature_vector[f'bin_{bin_label}_count'] = volume_bin_qty_sum.get(bin_label, 0)
    return feature_vector.fillna(0).to_list()

class ATCDataset_v2(Dataset):
    """
    Dataset class for ATC data. Input is created as feature vectors from the groups of items
    """
    def __init__(self, in_data_path, out_data_path):
        """
        Args:
            data_path (str): path to the data in csv format
        """
        self.in_df = pd.read_csv(
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
        
        # look for the used carton per the group, increase the number for proper box in the result vector
        for _, row in self.out_df.iterrows():
            cartonName = row['UsedCarton'].strip().upper()
            result_vector[row['GroupDelivery']][labels.index(cartonName)] += 1
        
        self.result_vector = result_vector
        
        # build INPUT DATA 
        # ========================================
        
        # Apply the function to each GroupDelivery
        def vectorize(group):
            return group[['X', 'Y', 'Z', 'Weight', 'Qty']].values.flatten().tolist()
        
        grouped = self.in_df.groupby('GroupDelivery').apply(create_feature_vector, include_groups=False).reset_index()

        # Rename columns
        grouped.columns = ['GroupDelivery', 'FeatureVector']
        
        # store vectors and group delivery
        self.feature_vectors = grouped.FeatureVector
        self.groupDelivery = grouped.GroupDelivery
        
    def __len__(self):
        return len(self.feature_vectors)
    
    def __getitem__(self, idx):
        gd = self.groupDelivery.iloc[idx]
        # Ensure the vectors are returned as numpy arrays
        input_vector = torch.tensor(self.feature_vectors.iloc[idx], dtype=torch.float32)  # Convert input vector to tensor
        output_vector = torch.tensor(self.result_vector[gd], dtype=torch.float32)  # Convert result vector to tensor
        return gd, input_vector, output_vector 