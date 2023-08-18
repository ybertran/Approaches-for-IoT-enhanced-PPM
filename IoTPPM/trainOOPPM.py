from __future__ import division, print_function

import copy
import csv
import os
import random
import sys
from collections import Counter
from math import log

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# torch packages
import torch.nn as nn
import unicodecsv
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from collections import Counter

pd.set_option("display.max_rows", None)
from collections import OrderedDict

from LSTM import Model
from CNN import CNNModel


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_prefix_data(data, min_length, max_length):
    # generate prefix data (each possible prefix becomes a trace)

    case_length = data.groupby(case_id_col)[activity_col].transform(len)
    data.loc[:, "case_length"] = case_length.copy()
    dt_prefixes = (
        data[data["case_length"] >= min_length].groupby(case_id_col).head(min_length)
    )
    dt_prefixes["prefix_nr"] = 1
    dt_prefixes["orig_case_id"] = dt_prefixes[case_id_col]
    for nr_events in range(min_length, max_length + 1):
        tmp = (
            data[data["case_length"] >= nr_events].groupby(case_id_col).head(nr_events)
        )
        tmp["orig_case_id"] = tmp[case_id_col]
        tmp[case_id_col] = tmp[case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
        tmp["prefix_nr"] = nr_events
        dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
    dt_prefixes["case_length"] = dt_prefixes["case_length"].apply(
        lambda x: min(max_length, x)
    )
    return dt_prefixes


def split_data_strict(data, train_ratio):
    # split into train and test using temporal split and discard events that overlap the periods
    data = data.sort_values(sorting_cols, ascending=True, kind="mergesort")
    grouped = data.groupby(case_id_col)
    start_timestamps = grouped[timestamp_col].min().reset_index()
    start_timestamps = start_timestamps.sort_values(
        timestamp_col, ascending=True, kind="mergesort"
    )
    train_ids = list(start_timestamps[case_id_col])[
        : int(train_ratio * len(start_timestamps))
    ]
    train = data[data[case_id_col].isin(train_ids)].sort_values(
        sorting_cols, ascending=True, kind="mergesort"
    )
    test = data[~data[case_id_col].isin(train_ids)].sort_values(
        sorting_cols, ascending=True, kind="mergesort"
    )
    split_ts = test[timestamp_col].min()
    train = train[train[timestamp_col] < split_ts]
    return (train, test)


def get_label_numeric(data, bins, labels):
    y = data.groupby(case_id_col).first()[label_col]  # one row per case

    label_mapping = {}
    for idx, (lower, upper) in enumerate(zip(bins, bins[1:])):
        label = labels[idx]
        label_mapping[label] = idx

    return [label_mapping[label] for label in y.values.tolist()]


def groupby_pad_all(train, test, val, cols, activity_col):
    activity_train, label_lists_train = groupby_pad(train, cols, activity_col)
    activity_test, label_lists_test = groupby_pad(test, cols, activity_col)
    activity_val, label_lists_val = groupby_pad(val, cols, activity_col)
    return (
        activity_train,
        activity_test,
        activity_val,
        label_lists_train,
        label_lists_test,
        label_lists_val,
    )


def groupby_pad(prefixes, cols, activity_col):
    ans_act, label_lists = groupby_caseID(prefixes, cols, activity_col)
    ######ACTIVITY########
    activity = pad_data(ans_act)
    return activity, label_lists


def pad_data(data):
    data[0] = nn.ConstantPad1d((0, max_prefix_length - data[0].shape[0]), 0)(data[0])
    padding = pad_sequence(data, batch_first=True, padding_value=0)
    return padding


def groupby_caseID(data, cols, col):
    groups = data[cols].groupby(case_id_col, as_index=True)
    # case_ids = groups.groups.keys()
    ans = [torch.tensor(list(y[col])) for _, y in groups]
    label_lists = [y[label_col].iloc[0] for _, y in groups]
    return ans, label_lists


def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = temp_list = log_df[log_df[column] != "none"][
        [column]
    ].values.tolist()  # remove all 'none' values from the index
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i
    # reorder by the index value
    alias = {k: v for k, v in sorted(alias.items(), key=lambda item: item[1])}
    return alias


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return torch.tensor(np.eye(num_classes, dtype="uint8")[y])


def create_indexes(i, data):
    dyn_index = create_index(data, i)
    index_dyn = {v: k for k, v in dyn_index.items()}
    dyn_weights = to_categorical(sorted(index_dyn.keys()), len(dyn_index))
    no_cols = len(data.groupby([i]))
    return dyn_weights, dyn_index, index_dyn, no_cols


def prepare_inputs(X_train, X_test):
    global ce
    ce = ColumnEncoder()
    X_train, X_test = X_train.astype(str), X_test.astype(str)
    X_train_enc = ce.fit_transform(X_train)
    X_test_enc = ce.transform(X_test)
    return X_train_enc, X_test_enc, ce


# https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2
class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None
        self.maps = dict()

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
            X_copy.loc[:, col] = X_copy.loc[:, col].apply(
                lambda x: self.maps[col].get(x, -1) + 1
            )
        return X_copy

    def get_maps(self):
        return self.maps

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            values = list(self.maps[col].keys())
            # find value in ordered list and map out of range values to None
            X_copy.loc[:, col] = [
                values[i - 1] if 0 < i <= len(values) else None for i in X_copy[col]
            ]
        return X_copy

    def fit(self, X, y=None):
        # only apply to string type columns
        self.columns = [col for col in X.columns if is_string_dtype(X[col])]
        for col in self.columns:
            self.maps[col] = OrderedDict(
                {value: num for num, value in enumerate(sorted(set(X[col])))}
            )
        return self


def to_categorical_all(train, test, val, num_classes):
    """1-hot encodes a tensor"""
    train_OHE = torch.tensor(np.eye(num_classes)[train])
    test_OHE = torch.tensor(np.eye(num_classes)[test])
    val_OHE = torch.tensor(np.eye(num_classes)[val])
    return train_OHE, test_OHE, val_OHE


def check_mode(MODE):
    if MODE == "LSTM":
        print("LSTM")
    elif MODE == "CNN":
        print("CNN")
    else:
        raise NotImplementedError(f"MODE {MODE} not implemented")


MODE = "LSTM"  # LSTM or CNN
check_mode(MODE)
case_id_col = "CaseID"
activity_col = "ActivityID"
timestamp_col = "CompleteTimestamp"
label_col = "Pump_Adjustment_Bin"
sorting_cols = [timestamp_col, activity_col]
cat_cols = [activity_col]
case_id_col = "CaseID"
activity_col = "ActivityID"
timestamp_col = "CompleteTimestamp"

min_prefix_length = 1
max_prefix_length = 80
batch_size = 512
learning_rate = 0.001
dropout = 0.1
lstm_size = 10
num_classes = 3
epochs = 50

# Define the bin edges and labels
bins = [0, 10, 25, 90]
labels = ["few", "medium", "many"]

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# read in original data
df = pd.read_csv("data/event_data_context_alex.csv", index_col="Unnamed: 0.1")
df = df.drop(["Unnamed: 0"], axis=1)

# Calculate the count of 'pump adjustment' activities for each case
pump_adjustment_counts = (
    df[df["ActivityID"] == "Pump adjustment"]
    .groupby("CaseID")
    .size()
    .reset_index(name="Pump_Adjustment_Count")
)

# Merge the counts back into the original DataFrame
df = df.merge(pump_adjustment_counts, on="CaseID", how="left")

# Fill NaN values with 0 for cases with no 'pump adjustment' activities
df["Pump_Adjustment_Count"].fillna(0, inplace=True)

train, test = split_data_strict(df, 0.8)
train, val = split_data_strict(train, 0.8)

no_cols_list = []
for i in cat_cols:
    _, _, _, no_cols = create_indexes(i, train)
    no_cols_list.append(no_cols)
vocab_size = [
    no_cols_list[0] + 1
]  # the vocabulary size is one bigger because you will add the padding value 0

dt_train_prefixes = generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_test_prefixes = generate_prefix_data(test, min_prefix_length, max_prefix_length)
dt_val_prefixes = generate_prefix_data(val, min_prefix_length, max_prefix_length)

# Calculate the count of 'pump adjustment' activities for each case
# train
pump_adjustment_counts = (
    dt_train_prefixes[dt_train_prefixes["ActivityID"] == "Pump adjustment"]
    .groupby("CaseID")
    .size()
    .reset_index(name="Pump_Adjustment_Count_Prefixes")
)
dt_train_prefixes = dt_train_prefixes.merge(
    pump_adjustment_counts, on="CaseID", how="left"
)
dt_train_prefixes["Pump_Adjustment_Count_Prefixes"].fillna(0, inplace=True)
dt_train_prefixes["Remaining_Adjustment"] = (
    dt_train_prefixes["Pump_Adjustment_Count"]
    - dt_train_prefixes["Pump_Adjustment_Count_Prefixes"]
)
# test
pump_adjustment_counts = (
    dt_test_prefixes[dt_test_prefixes["ActivityID"] == "Pump adjustment"]
    .groupby("CaseID")
    .size()
    .reset_index(name="Pump_Adjustment_Count_Prefixes")
)
dt_test_prefixes = dt_test_prefixes.merge(
    pump_adjustment_counts, on="CaseID", how="left"
)
dt_test_prefixes["Pump_Adjustment_Count_Prefixes"].fillna(0, inplace=True)
dt_test_prefixes["Remaining_Adjustment"] = (
    dt_test_prefixes["Pump_Adjustment_Count"]
    - dt_test_prefixes["Pump_Adjustment_Count_Prefixes"]
)
# val
pump_adjustment_counts = (
    dt_val_prefixes[dt_val_prefixes["ActivityID"] == "Pump adjustment"]
    .groupby("CaseID")
    .size()
    .reset_index(name="Pump_Adjustment_Count_Prefixes")
)
dt_val_prefixes = dt_val_prefixes.merge(pump_adjustment_counts, on="CaseID", how="left")
dt_val_prefixes["Pump_Adjustment_Count_Prefixes"].fillna(0, inplace=True)
dt_val_prefixes["Remaining_Adjustment"] = (
    dt_val_prefixes["Pump_Adjustment_Count"]
    - dt_val_prefixes["Pump_Adjustment_Count_Prefixes"]
)

# Create a new column with the bin labels
dt_train_prefixes["Pump_Adjustment_Bin"] = pd.cut(
    dt_train_prefixes["Remaining_Adjustment"], bins=bins, labels=labels, right=False
)
dt_test_prefixes["Pump_Adjustment_Bin"] = pd.cut(
    dt_test_prefixes["Remaining_Adjustment"], bins=bins, labels=labels, right=False
)
dt_val_prefixes["Pump_Adjustment_Bin"] = pd.cut(
    dt_val_prefixes["Remaining_Adjustment"], bins=bins, labels=labels, right=False
)

train_y = get_label_numeric(dt_train_prefixes, bins, labels)
test_y = get_label_numeric(dt_test_prefixes, bins, labels)
val_y = get_label_numeric(dt_val_prefixes, bins, labels)

label_counts_train = Counter(train_y)
label_counts_test = Counter(test_y)

print("distribution of labels:")
print("train")
# Print the count of each label
for label, count in label_counts_train.items():
    print(f"Label {label}: Count {count}")
print("test")
# Print the count of each label
for label, count in label_counts_test.items():
    print(f"Label {label}: Count {count}")

# prepare inputs
train_cat_cols, test_cat_cols, ce = prepare_inputs(
    dt_train_prefixes.loc[:, cat_cols], dt_test_prefixes.loc[:, cat_cols]
)
train_cat_cols, val_cat_cols, _ = prepare_inputs(
    dt_train_prefixes.loc[:, cat_cols], dt_val_prefixes.loc[:, cat_cols]
)

dt_train_prefixes[cat_cols] = train_cat_cols
dt_test_prefixes[cat_cols] = test_cat_cols
dt_val_prefixes[cat_cols] = val_cat_cols

# groupby case ID
cols = list(dt_train_prefixes.columns)
(
    activity_train,
    activity_test,
    activity_val,
    label_lists_train,
    label_lists_test,
    label_lists_val,
) = groupby_pad_all(
    dt_train_prefixes, dt_test_prefixes, dt_val_prefixes, cols, activity_col
)
# OHE all
activity_train_OHE, activity_test_OHE, activity_val_OHE = to_categorical_all(
    activity_train, activity_test, activity_val, vocab_size[0]
)

preds_all = []
test_y_all = []
score = 0

dataset = torch.utils.data.TensorDataset(activity_train_OHE)
dataset = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    worker_init_fn=seed_worker,
)
if MODE == "LSTM":
    model = Model(vocab_size, dropout, lstm_size, num_classes).to(device)
elif MODE == "CNN":
    model = CNNModel(vocab_size, num_classes).to(device)

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

lr_reducer = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    verbose=False,
    threshold=0.0001,
    cooldown=0,
    min_lr=0,
)

criterion = nn.CrossEntropyLoss()
print("training")
best_acc = 0

for epoch in range(epochs):
    print("Epoch: ", epoch)
    for i, data_act in enumerate(
        dataset, 0
    ):  # loop over the data, and jump with step = bptt.
        model.train()
        data_act = data_act.to(device)
        y_ = model(data_act).to(device)
        y_probs = F.softmax(y_, dim=1)  # Apply softmax activation
        train_batch = torch.tensor(
            train_y[i * batch_size : (i + 1) * batch_size], dtype=torch.long
        ).to(
            device
        )  # Change to long data type
        train_loss = criterion(y_probs, train_batch)
        optimizer.zero_grad()
        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    print("train loss", train_loss)
with torch.no_grad():
    model.eval()
    print("testing")
    pred = model(activity_test_OHE).squeeze(-1).to(device)
    pred = F.softmax(pred, dim=1)  # Apply softmax activation
    pred = pred.cpu()

pred_indices = torch.argmax(
    pred, dim=1
)  # Convert predicted probabilities to class predictions
# Convert test_y to a PyTorch tensor
test_y_tensor = torch.tensor(test_y, dtype=torch.long).to(device)
print("pred", pred_indices)
print("test y", test_y)
correct_predictions = (
    (pred_indices == test_y_tensor).sum().item()
)  # Count correct predictions
total_examples = len(test_y_tensor)
accuracy = correct_predictions / total_examples
print("Accuracy:", accuracy)

label_counts_pred = Counter(list(pred_indices))

# Print the count of each label
for label, count in label_counts_pred.items():
    print(f"Label {label}: Count {count}")
