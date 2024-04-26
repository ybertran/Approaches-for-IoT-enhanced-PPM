###### IMPORT FUNCTIONS
from __future__ import division, print_function
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from collections import Counter
#own util function data creation
from util.DataCreation import DataCreation
from LSTM import Model
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

#DATASET IDENTIFIERS AND PARAMETERS
case_id_col = "CaseID"
activity_col = "ActivityID"
timestamp_col = "CompleteTimestamp"
label_col = "Pump_Adjustment_Bin"

#LSTM NEURAL NETWORK PARAMETERS
batch_size = 64
learning_rate = 0.001
dropout = 0.1
lstm_size = 2
epochs = 200

#read in original data
df = pd.read_csv('data/event_data_aggregated_alex2.csv', index_col="Unnamed: 0")
print('the shape of the dataframe', df.shape)

#PREPROCESSING CHOICES
train_ratio = 0.8
min_prefix_length = 1

print(df.dtypes)

#create the numerical features

numerical_features_col = list(df.columns)
numerical_features_col.remove('CaseID')
numerical_features_col.remove('CompleteTimestamp')
numerical_features_col.remove('Vessel')
numerical_features_col.remove('event_id')
numerical_features_col.remove('ActivityID')
#numerical_features_col.remove('lifecycle:transition')
#numerical_features_col.remove('Pump Circulation Flow_sum')
#numerical_features_col.remove('Filter 1 DeltaP_sum')
#numerical_features_col.remove('Filter 2 DeltaP_sum')
#numerical_features_col.remove('Tank Pressure_sum')
#numerical_features_col.remove('Filter 1 Inlet Pressure_sum')

#numerical_features_col = ['Filter 1 DeltaP_mean', 'Filter 2 DeltaP_mean']

#datacreator
datacreator = DataCreation(train_ratio, case_id_col, activity_col, timestamp_col, label_col, numerical_features_col)

# Calculate the count of 'pump adjustment' activities for each case
pump_adjustment_counts = df[df['ActivityID'] == 'Pump adjustment'].groupby('CaseID').size().reset_index(name='Pump_Adjustment_Count')
df = df.merge(pump_adjustment_counts, on='CaseID', how='left')
#pd.set_option('display.max_rows', None)

# Fill NaN values with 0 for cases with no 'pump adjustment' activities
df['Pump_Adjustment_Count'].fillna(0, inplace=True)

#out-of-time split into train and test to avoid data leakage. Next, we take a part of the training data for validation purposes 
# Min-max normalization feature-wise
#for col in numerical_features_col:
#    min_val = df[col].min()
#    max_val = df[col].max()
#    df[col] = (df[col] - min_val) / (max_val - min_val)

train, test = datacreator.split_data_strict(df)
_, val = datacreator.split_data_strict(train)

#to generate the vocabulary size. This is necessary for the LSTM architecture
vocab_size = datacreator.vocabulary_size(df)

#generate prefix cases. This takes the prefixes of each case, i.e. a case [a,b,a,d,c] of length 5 has prefixes [a,b,a,d,c], [a,b,a,d], [a,b,a], [a,b], [a] 
dt_train_prefixes = datacreator.generate_prefix_data(train, min_prefix_length, 'train')
dt_test_prefixes = datacreator.generate_prefix_data(test, min_prefix_length)
dt_val_prefixes = datacreator.generate_prefix_data(val, min_prefix_length)

# Calculate the count of 'pump adjustment' activities for each case. This is different to the count of the total case, and we need to calculate this to obtain the "remaining count of pump adjustments"
# the feature 'CaseID' refers to the case ID of the prefixes, and the feature 'orig_case_id' to the case ID of the original trace
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
bins = [0,13,28,133]
labels = [int(0), int(1), int(2)]
num_classes = 3

# Create a new column with the bin labels
dt_train_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_train_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)
dt_test_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_test_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)
dt_val_prefixes['Pump_Adjustment_Bin'] = pd.cut(dt_val_prefixes['Remaining_Adjustment'], bins=bins, labels=labels, right=False)

#extract the label of the prefixes
train_y = datacreator.get_label_numeric(dt_train_prefixes, bins, labels)
test_y = datacreator.get_label_numeric(dt_test_prefixes, bins, labels)
val_y = datacreator.get_label_numeric(dt_val_prefixes, bins, labels)

#TO CHECK WHETHER THERE IS CLASS IMBALANCE
label_counts_train = Counter(train_y)
label_counts_val = Counter(val_y)
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

print('val')
# Print the count of each label
for label, count in label_counts_val.items():
    print(f"Label {label}: Count {count}")

print('undersample train')
dt_train_prefixes = datacreator.undersample_cases(dt_train_prefixes) #undersample the train cases

print('undersample test')
dt_test_prefixes = datacreator.undersample_cases(dt_test_prefixes) #undersample the test cases

print('undersample val')
dt_val_prefixes = datacreator.undersample_cases(dt_val_prefixes) #undersample the test cases

#extract the label of the prefixes
train_y = datacreator.get_label_numeric(dt_train_prefixes, bins, labels)
test_y = datacreator.get_label_numeric(dt_test_prefixes, bins, labels)
val_y = datacreator.get_label_numeric(dt_val_prefixes, bins, labels)

#TO CHECK WHETHER THERE IS CLASS IMBALANCE
label_counts_train = Counter(train_y)
label_counts_test = Counter(test_y)
label_counts_val = Counter(val_y)

print('distribution of labels (after):')
print('train')
# Print the count of each label
for label, count in label_counts_train.items():
    print(f"Label {label}: Count {count}")

print('val')
# Print the count of each label
for label, count in label_counts_val.items():
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

cols = list(dt_train_prefixes.columns)
print(dt_train_prefixes)

activity_train, activity_test, activity_val, train_y, test_y, val_y = datacreator.groupby_pad_all(dt_train_prefixes, dt_test_prefixes, dt_val_prefixes, cols, activity_col)

# Binary values for the labels
# Create three labels based on conditions

numeric_train, numeric_test, numeric_val = datacreator.groupby_pad_all_num(dt_train_prefixes, dt_test_prefixes, dt_val_prefixes, numerical_features_col)

"""
Indices of NaN values: tensor([[3886,   45,    0],
        [3886,   45,    1],
        [3886,   46,    0],
        ...,
        [4381,    7,    1],
        [4381,    8,    0],
        [4381,    8,    1]])



nan_indices = torch.nonzero(torch.isnan(numeric_train))

# Print the indices where NaN values are True
print("Indices of NaN values:", nan_indices)

if torch.any(result1):
    print('this')

elif torch.any(result2):
    print('that')
"""

print(activity_train.shape, numeric_train.shape)

dataset = torch.utils.data.TensorDataset(activity_train, numeric_train)
dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

preds_all = []
test_y_all = []
score = 0

model = Model(vocab_size, len(numerical_features_col), dropout, lstm_size, num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
            
lr_reducer = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, 
                                threshold=0.0001, cooldown=0, min_lr=0)

print('training')
best_acc = 0

for epoch in range(epochs):
        print("Epoch: ", epoch)
        #for i, data_act in enumerate(dataset, 0):
        for i, (data_act, data_num) in enumerate(dataset, 0):
            model.train()
            data_act = data_act.to(device)
            data_num = data_num.to(device)
            y_ = model(data_act, data_num, 'training').to(device) 
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_batch = torch.tensor(train_y[i*batch_size:(i+1)*batch_size], dtype=torch.long).to(device)  # Change to long data type
            log_probs = F.log_softmax(y_, dim=1)
            loss = F.nll_loss(log_probs, train_batch)
            #loss = F.cross_entropy(y_, train_batch)
            loss.backward()
            optimizer.step()
        print('train loss', loss)
        with torch.no_grad():
            model.eval()
            print('testing')
            pred = model(activity_train, numeric_train, 'testing').squeeze(-1).to(device)
            pred = pred.cpu()
            pred_indices = torch.argmax(pred, dim=1)  # Convert predicted probabilities to class predictions
            test_y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)  # Convert test_y to a PyTorch tensor
            correct_predictions = (pred_indices == test_y_tensor).sum().item()  # Count correct predictions
            total_examples = len(test_y_tensor)
            accuracy = correct_predictions / total_examples
            print("Accuracy:", accuracy)