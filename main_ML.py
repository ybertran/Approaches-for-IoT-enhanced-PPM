###### IMPORT FUNCTIONS
from __future__ import division, print_function
from collections import Counter
from math import log
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import is_string_dtype
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from collections import Counter
#own util function data creation
from util.DataCreation import DataCreation
from LSTM import Model
import sklearn
import matplotlib.pyplot as plt
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

#DATASET IDENTIFIERS AND PARAMETERS
case_id_col = "CaseID"
activity_col = "ActivityID"
timestamp_col = "CompleteTimestamp"
label_col = "Pump_Adjustment_Bin"

#LSTM NEURAL NETWORK PARAMETERS
batch_size = 512
learning_rate = 0.001
dropout = 0.3
lstm_size = 10
epochs = 100

#read in original data
#df = pd.read_csv('data/event_data_context_alex2.csv', index_col ="Unnamed: 0")
df = pd.read_csv()

print('the shape of the dataframe', df.shape)

#PREPROCESSING CHOICES
train_ratio = 0.8
min_prefix_length = 1
max_prefix_length = 132

# encodings dictionary
encoding = ['agg']
encoding_dict = {"agg": ["agg"]}

#create the numerical features
numerical_features_col = list(df.columns)
numerical_features_col.remove('CaseID')
numerical_features_col.remove('CompleteTimestamp')
numerical_features_col.remove('Vessel')
numerical_features_col.remove('event_id')
numerical_features_col.remove('ActivityID')
numerical_features_col.remove('lifecycle:transition')

#datacreator
datacreator = DataCreation(train_ratio, case_id_col, activity_col, timestamp_col, label_col, numerical_features_col)

# Calculate the count of 'pump adjustment' activities for each case
pump_adjustment_counts = df[df['ActivityID'] == 'Pump adjustment'].groupby('CaseID').size().reset_index(name='Pump_Adjustment_Count')
df = df.merge(pump_adjustment_counts, on='CaseID', how='left')

# Fill NaN values with 0 for cases with no 'pump adjustment' activities
df['Pump_Adjustment_Count'].fillna(0, inplace=True)

# Calculate the frequency of 'Pump_Adjustment_Count' values across cases
case_frequency_counts = df.groupby('Pump_Adjustment_Count')['CaseID'].nunique()

#out-of-time split into train and test to avoid data leakage
train, test = datacreator.split_data_strict(df)
train, val = datacreator.split_data_strict(train)

#to generate the vocabulary size. This is necessary for the LSTM architecture
vocab_size = datacreator.vocabulary_size(df)

#generate prefix cases. This takes the prefixes of each case, i.e. a case [a,b,a,d,c] of length 5 has prefixes [a,b,a,d,c], [a,b,a,d], [a,b,a], [a,b], [a] 
dt_train_prefixes = datacreator.generate_prefix_data(train, min_prefix_length, 'train')
dt_test_prefixes = datacreator.generate_prefix_data(test, min_prefix_length)
dt_val_prefixes = datacreator.generate_prefix_data(val, min_prefix_length)

# Calculate the count of 'pump adjustment' activities for each case. This is different to the count of the total case, and we need to calculate this to obtain the "remaining count of pump adjustments"
#train
pump_adjustment_counts = dt_train_prefixes[dt_train_prefixes['ActivityID'] == 'Pump adjustment'].groupby('CaseID').size().reset_index(name='Pump_Adjustment_Count_Prefixes')
dt_train_prefixes = dt_train_prefixes.merge(pump_adjustment_counts, on='CaseID')
dt_train_prefixes['Pump_Adjustment_Count_Prefixes'].fillna(0, inplace=True)
dt_train_prefixes['Remaining_Adjustment'] = dt_train_prefixes['Pump_Adjustment_Count'] - dt_train_prefixes['Pump_Adjustment_Count_Prefixes']

#test
pump_adjustment_counts = dt_test_prefixes[dt_test_prefixes['ActivityID'] == 'Pump adjustment'].groupby('CaseID').size().reset_index(name='Pump_Adjustment_Count_Prefixes')
dt_test_prefixes = dt_test_prefixes.merge(pump_adjustment_counts, on='CaseID', how='left')
dt_test_prefixes['Pump_Adjustment_Count_Prefixes'].fillna(0, inplace=True)
dt_test_prefixes['Remaining_Adjustment'] = dt_test_prefixes['Pump_Adjustment_Count'] - dt_test_prefixes['Pump_Adjustment_Count_Prefixes']
#val
pump_adjustment_counts = dt_val_prefixes[dt_val_prefixes['ActivityID'] == 'Pump adjustment'].groupby('CaseID').size().reset_index(name='Pump_Adjustment_Count_Prefixes')
dt_val_prefixes = dt_val_prefixes.merge(pump_adjustment_counts, on='CaseID', how='left')
dt_val_prefixes['Pump_Adjustment_Count_Prefixes'].fillna(0, inplace=True)
dt_val_prefixes['Remaining_Adjustment'] = dt_val_prefixes['Pump_Adjustment_Count'] - dt_val_prefixes['Pump_Adjustment_Count_Prefixes']


# Define the bin edges and labels
bins = [0,15,30,80]
labels = ['few', 'medium', 'many']
num_classes = 3

# Create a new column with the bin labels
dt_train_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_train_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)
dt_test_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_test_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)
dt_val_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_val_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)

# groupby case ID
cols = list(dt_train_prefixes.columns)

#extract the label of the prefixes
train_y = datacreator.get_label_numeric(dt_train_prefixes, bins, labels)
test_y = datacreator.get_label_numeric(dt_test_prefixes, bins, labels)
val_y = datacreator.get_label_numeric(dt_val_prefixes, bins, labels)

#TO CHECK WHETHER THERE IS CLASS IMBALANCE
label_counts_train = Counter(train_y)
label_counts_test = Counter(test_y)

print('distribution of labels (before):')
print('train')
# Print the count of each label
for label, count in label_counts_train.items():
    print(f"Label {label}: Count {count}")
print('test')
# Print the count of each label
for label, count in label_counts_test.items():
    print(f"Label {label}: Count {count}")

dt_train_prefixes = datacreator.undersample_cases(dt_train_prefixes) #undersample the train cases
dt_val_prefixes = datacreator.undersample_cases(dt_val_prefixes) #undersample the test cases
dt_test_prefixes = datacreator.undersample_cases(dt_test_prefixes) #undersample the test cases

#extract the label of the prefixes
train_y = datacreator.get_label_numeric(dt_train_prefixes, bins, labels)
test_y = datacreator.get_label_numeric(dt_test_prefixes, bins, labels)
val_y = datacreator.get_label_numeric(dt_val_prefixes, bins, labels)

#TO CHECK WHETHER THERE IS CLASS IMBALANCE
label_counts_train = Counter(train_y)
label_counts_val = Counter(val_y)
label_counts_test = Counter(test_y)

print('distribution of labels (after):')
print('train')
# Print the count of each label
for label, count in label_counts_train.items():
    print(f"Label {label}: Count {count}")
print('test')
# Print the count of each label
for label, count in label_counts_test.items():
    print(f"Label {label}: Count {count}")

#prepare inputs
train_cat_cols, test_cat_cols, ce = datacreator.prepare_inputs(dt_train_prefixes.loc[:, [activity_col]], dt_test_prefixes.loc[:, [activity_col]])
train_cat_cols, val_cat_cols, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:, [activity_col]], dt_val_prefixes.loc[:, [activity_col]])
dt_train_prefixes[activity_col] = train_cat_cols
dt_test_prefixes[activity_col] = test_cat_cols
dt_val_prefixes[activity_col] = val_cat_cols

activity_train, activity_test, activity_val, label_lists_train, label_lists_test, label_lists_val = datacreator.groupby_pad_all(dt_train_prefixes, dt_test_prefixes, dt_val_prefixes, cols, activity_col)
activity_train = activity_train.numpy()
print('the prefixes')
print(activity_train)

import numpy as np
# Calculate vocabulary size
vocab_size = np.max(activity_train) + 1

# Create an empty dataframe
df_train = pd.DataFrame(np.zeros((activity_train.shape[0], vocab_size), dtype=int))
df_test = pd.DataFrame(np.zeros((activity_test.shape[0], vocab_size), dtype=int))

# Count frequency of values in each row and update the dataframe
for i, row in enumerate(activity_train):
    unique, counts = np.unique(row, return_counts=True)
    df_train.loc[i, unique] = counts

# Count frequency of values in each row and update the dataframe
for i, row in enumerate(activity_test):
    unique, counts = np.unique(row, return_counts=True)
    df_test.loc[i, unique] = counts

# Remove the first column
df_train = df_train.drop(df_train.columns[0], axis=1)
df_test = df_test.drop(df_test.columns[0], axis=1)

print("DataFrame with Frequency Counts:")
print(df_train)

import xgboost as xgb

cls = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500)

cls.fit(df_train, train_y)
                
#predictions

pred = cls.predict_proba(df_test)
print('the probabilities for each label:')
print(pred)
pred_indices = torch.tensor(np.argmax(pred, axis=1)) # Convert predicted probabilities to class predictions
# Convert test_y to a PyTorch tensor
test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
correct_predictions = (pred_indices == test_y_tensor).sum().item()  # Count correct predictions
total_examples = len(test_y_tensor)
accuracy = correct_predictions / total_examples
print("Accuracy:", accuracy)
