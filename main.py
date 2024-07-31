### Main script to run the training and testing of the model
### Author: Jan Holan
### Date: 2024-07-31

import atc_dataloader, atc_model


def train_model(model, dataset_train, dataset_val, epochs=10):
    pass

def test_model(model, dataset_input, dataset_val):    
    pass

if __name__ == "__main__":
    # load data
    PATH_INPUT_DATA="data/atc_data.csv"
    PATH_RESULTS="data/results.csv"
    
    atc_data = atc_dataloader.ATCDataset(data_path=PATH_INPUT_DATA)
