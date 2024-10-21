# IOTPPM

The file to train the LSTM:
- main_LSTM.py

In this file, we read in the data, preprocess it and train the model. The data is read in from a CSV file, which is then preprocessed.

- evaluate_NAP.py

This file appears is part of the project that evaluates the LSTM model performance on a dataset of event logs. 
It loads the pre-trained Keras model from a specified file path and reads a CSV file containing event log data into a Pandas DataFrame.

## util
Contains the files:
- Arguments.py # Args class for handling dataset-specific arguments and parameters.
- dataset_confs.py # Class to handle the dataset configurations
- dataset_manager.py # Class to handle the dataset
- EncoderFactory.py # Class to handle the encoder



## other
contains license, readme, .gitignore. Feel free to make changes on anything.