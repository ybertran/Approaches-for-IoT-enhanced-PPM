# IOTPPM

The file to train the XGBoost baseline:
- IoT_baseline.py

In this file, we read in the data as CSV files, preprocess them and train the model.

- evaluate_XGB.py

In this file, we evaluate the model, which is read in as a JSON file containing all the parameters and it is run on a test set. 
The various metrics used to compare models are computed.

The file to train the LSTM:
- main_LSTM.py

In this file, we read in the data, preprocess it and train the model. The data is read in from a CSV file, which is then preprocessed.

- evaluate_NAP.py

This file appears is part of the project that evaluates the LSTM model performance on a dataset of event logs. 
It loads the pre-trained Keras model from a specified file path and reads a CSV file containing event log data into a Pandas DataFrame.

- requirements.txt

The requirements for Python and the packages

## util
Contains the files:
- Arguments.py # Args class for handling dataset-specific arguments and parameters.
- dataset_confs.py # Class to handle the dataset configurations
- dataset_manager.py # Class to handle the dataset
- EncoderFactory.py # Class to handle the encoder

## other
contains license, readme, .gitignore. Feel free to make changes on anything.
