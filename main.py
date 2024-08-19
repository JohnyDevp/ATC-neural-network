### Main script to run the training and testing of the model
### Author: Jan Holan
### Date: 2024-07-31

import atc_dataloader, atc_model
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(model, dataset_train, dataset_val, optimizer, epochs=10):
    
    loss = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        # set the model to train mode
        model.train()
        for batch in dataset_train:
            # get the data from the batch
            data = batch['data'].to(device)
            target = batch['target'].to(device)
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward pass
    

def test_model(model, dataset_input, dataset_val):    
    pass

if __name__ == "__main__":
    # load data
    PATH_INPUT_DATA="data/nn_input_data.csv"
    PATH_RESULTS="data/results.csv"
    
    # atc_data = atc_dataloader.ATCDataset(data_path=PATH_INPUT_DATA)
    atc_data = pd.read_csv(PATH_INPUT_DATA, delimiter=",", header=0)
    result_data =pd.read_csv(
        PATH_RESULTS,
        delimiter=",",
        header=0)
   
    # labels for vector of the ouput boxes 
    labels = [
        "OM0000", "OM0001", "OM0002", "OM0003", "OM0004", "OM0005", "OM0006", "OM0007", 
        "OM0008", "OM0009", "OM0010", "PAL01", "PAL02", "PAL03", "PAL04", "PAL05", "PAL06", 
        "PAL07"
    ]

    # result vector contains number of occurences of each box for each group of items
    result_vector = {}
    for group in result_data['GroupDelivery'].unique():
        result_vector[group] = np.zeros((labels.__len__()))
        
    for _, row in result_data.iterrows():
        cartonName = row['UsedCarton'].strip().upper()
        result_vector[row['GroupDelivery']][labels.index(cartonName)] += 1

    # build input data
    df = atc_data

    # Function to flatten and pad the vector
    def create_vector(group, target_length=5000):
        # Flatten the values into a single list
        vector = group[['X', 'Y', 'Z', 'Weight', 'Qty']].values.flatten().tolist()
        
        # Pad with zeros or truncate to fit the target length
        if len(vector) < target_length:
            vector += [0] * (target_length - len(vector))
        else:
            vector = vector[:target_length]
        
        return vector

    # Apply the function to each GroupDelivery
    grouped = df.groupby('GroupDelivery').apply(create_vector, include_groups=False).reset_index()

    # Rename columns
    grouped.columns = ['GroupDelivery', 'Vector']

    # Display the result
    print(grouped.where(grouped['GroupDelivery'] == 11862489).dropna())