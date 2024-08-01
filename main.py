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


def train_model(model, dataset_train, dataset_val, epochs=10):
    pass

def test_model(model, dataset_input, dataset_val):    
    pass

if __name__ == "__main__":
    # load data
    PATH_INPUT_DATA="data/nn_input_data.csv"
    PATH_RESULTS="data/results.csv"
    
    atc_data = atc_dataloader.ATCDataset(data_path=PATH_INPUT_DATA)
    result_data =pd.read_csv(
        PATH_RESULTS,
        delimiter=",",
        header=0)
    
    labels = [
        "OM0000", "OM0001", "OM0002", "OM0003", "OM0004", "OM0005", "OM0006", "OM0007", 
        "OM0008", "OM0009", "OM0010", "PAL01", "PAL02", "PAL03", "PAL04", "PAL05", "PAL06", 
        "PAL07"
    ]

    result_vector = {} #np.zeros((1000, labels.__len__()))
    for group in result_data['GroupDelivery'].unique():
        result_vector[group] = np.zeros((labels.__len__()))
        
    for _, row in result_data.iterrows():
        result_vector[row['GroupDelivery']][labels.index(row['UsedCarton'].strip().upper())] += 1

    print(result_vector)