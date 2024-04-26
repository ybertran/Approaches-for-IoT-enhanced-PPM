from __future__ import division, print_function

import numpy as np
import pandas as pd
import torch
# torch packages
import torch.nn as nn
from pandas.api.types import is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from sklearn.pipeline import FeatureUnion
import util.EncoderFactory as EncoderFactory
from sklearn.preprocessing import MinMaxScaler

class DataCreation:
    """preprocessing steps"""

    def __init__(self, train_ratio, case_id_col, activity_col, timestamp_col, label_col, numerical_features_col):
        self.train_ratio = train_ratio
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.sorting_cols = [self.timestamp_col, self.activity_col]
        self.label_col = label_col
        self.numerical_features_col = numerical_features_col
        self.encoding_dict = {"agg": ["agg"]}
        self.cls_encoding ='agg'

    def generate_prefix_data(self,data, min_length, mode=None):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)
        if mode =='train':
            self.max_prefix_length = data['case_length'].max()
        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length, self.max_prefix_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(self.max_prefix_length, x))
        return dt_prefixes

    def split_data_strict(self, data):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(self.train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)

    def get_label_numeric(self, data, bins, labels):
        y = data.groupby(self.case_id_col).first()[self.label_col]  # one row per case
        
        label_mapping = {}
        for idx, (lower, upper) in enumerate(zip(bins, bins[1:])):
            label = labels[idx]
            label_mapping[label] = idx
        
        return [label_mapping[label] for label in y.values.tolist()]


    def groupby_pad_all(self, train, test, val, cols, activity_col):
        activity_train, label_lists_train = self.groupby_pad(train, cols, activity_col)
        activity_test,label_lists_test = self.groupby_pad(test, cols, activity_col)
        activity_val, label_lists_val = self.groupby_pad(val, cols, activity_col)
        return activity_train, activity_test, activity_val, label_lists_train, label_lists_test, label_lists_val

    def groupby_pad(self, prefixes, cols, activity_col):
        ans_act, label_lists, case_ids = self.groupby_caseID(prefixes, cols, activity_col)
        ######ACTIVITY########
        activity = self.pad_data(ans_act)
        return activity, label_lists
 #data[0] = nn.ConstantPad1d((0, self.max_prefix_length - data[0].shape[0]), 0)(data[0])
    def pad_data(self, data):
        padded_list = [lst + [0] * (self.max_prefix_length - len(lst)) for lst in data]
        tensor_list = [torch.tensor(lst) for lst in padded_list]
       
        padding = pad_sequence(tensor_list, batch_first=True, padding_value=0)
        return padding
    
    def is_ordered(self, lst):
        return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

    def groupby_caseID(self, data, cols, col):
        # Initialize lists to store sequences and labels for 'deviant' and 'regular' cases
        groups = data[cols].groupby('CaseID', as_index=False)
        groups = groups.apply(lambda group: group.sort_values('event_id'))
        grouped = groups.groupby('CaseID')
        sequences = []
        labels = []
        case_ids = []
        for case_id, group in grouped:
            if self.is_ordered(list(group['event_id'])): 
                label = group['Pump_Adjustment_Bin'].iloc[0]
                sequence = list(group[col])
                sequences.append(sequence)
                labels.append(label)
                case_ids.append(case_id)
            else:
                print('problem', group)
                breakpoint()
                AssertionError
        # Concatenate sequences and labels for 'deviant' and 'regular' cases
        return sequences, labels, case_ids
    """
    def groupby_caseID(self, data, cols, col):
        groups = data[cols].groupby(self.case_id_col, as_index=True)
        #case_ids = groups.groups.keys()
        ans = [torch.tensor(list(y[col])) for _, y in groups]
        label_lists = [y[self.label_col].iloc[0] for _, y in groups]
        return ans, label_lists
    
    def groupby_caseID_num(self, data, cols):
        groups = data.groupby(self.case_id_col, as_index=True)
        ans_num = [torch.tensor(y[self.numerical_features_col].values, dtype=torch.long) for _, y in groups]
        return ans_num
    """
    def groupby_caseID_num(self, data, cols):
        groups = data.groupby(self.case_id_col, as_index=True)

        ans_padded = []
        for _, y in groups:
            group_data = {col: torch.tensor(list(y[col])) for col in cols}

            # Pad sequences with zeros to the maximum length
            padded_seq = {}
            for col, values in group_data.items():
                padded_values = np.pad(values.numpy(), (0, self.max_prefix_length - len(values)), 'constant')
                padded_seq[col] = torch.tensor(padded_values)

            ans_padded.append(padded_seq)

        # Convert the list of dictionaries to a 3D tensor
        batch_size = len(ans_padded)
        features = len(cols)
        seq_length = self.max_prefix_length

        result_array = torch.zeros((batch_size, seq_length, features), dtype=torch.float32)
        for i, seq_dict in enumerate(ans_padded):
            for j, (col, values) in enumerate(seq_dict.items()):
                result_array[i, :len(values), j] = values

        return result_array

    def groupby_pad_all_num(self, train, test, val, cols):
        numerical_features_train = self.groupby_pad_num(train, cols)
        numerical_features_test = self.groupby_pad_num(test, cols)
        numerical_features_val = self.groupby_pad_num(val, cols)
        return numerical_features_train, numerical_features_test, numerical_features_val

    def pad_data_num(self, data):
        padded_data = [nn.ConstantPad2d((0, 0, 0,  self.max_prefix_length - seq.shape[0]), 0)(seq) for seq in data]
        padding = torch.stack(padded_data, dim=0)
        return padding

    def groupby_pad_num(self, prefixes, cols):
        ans_num = self.groupby_caseID_num(prefixes, cols)
       
        # Identify NaN values using torch.isnan
        nan_mask = torch.isnan(ans_num)

        # Replace NaN values with zero using boolean indexing
        tensor_no_nan = ans_num.clone()
        tensor_no_nan[nan_mask] = 0.0

        return tensor_no_nan

    def create_index(self, log_df, column):
        """Creates an idx for a categorical attribute.
        Args:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
        """
        temp_list = temp_list = log_df[log_df[column] != 'none'][[column]].values.tolist()  # remove all 'none' values from the index
        subsec_set = {(x[0]) for x in temp_list}
        subsec_set = sorted(list(subsec_set))
        alias = dict()
        for i, _ in enumerate(subsec_set):
            alias[subsec_set[i]] = i
        # reorder by the index value
        alias = {k: v for k, v in sorted(alias.items(), key=lambda item: item[1])}
        return alias

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return torch.tensor(np.eye(num_classes, dtype='uint8')[y])

    def create_indexes(self, i, data):
        dyn_index = self.create_index(data, i)
        index_dyn = {v: k for k, v in dyn_index.items()}
        dyn_weights = self.to_categorical(sorted(index_dyn.keys()), len(dyn_index))
        no_cols = len(data.groupby([i]))
        return dyn_weights,  dyn_index, index_dyn, no_cols

    def prepare_inputs(self, X_train, X_test):
            global ce
            ce = self.ColumnEncoder()
            X_train, X_test = X_train.astype(str), X_test.astype(str)
            X_train_enc = ce.fit_transform(X_train)
            X_test_enc = ce.transform(X_test)
            return X_train_enc, X_test_enc, ce
    
    def undersample_cases(self, data):
        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=[self.case_id_col, self.label_col])
        label_counts = unique_cases[self.label_col].value_counts() # Calculate class distribution

        underrepresented_label = label_counts.idxmin() #the least represented labels
        overrepresented_labels = set(label_counts.index)
        overrepresented_labels.remove(underrepresented_label)
        #overrepresented_label = label_counts.idxmax() # Find the labels with their counts
        
        num_cases_to_keep = label_counts.min() # Count of cases to keep for each class (matching the most underrepresented class)
        for overrepresented_label in overrepresented_labels:
            the_other_label = [num for num in overrepresented_labels if num != overrepresented_label][0]
            overrepresented_case_ids = data[data[self.label_col] == overrepresented_label][self.case_id_col].unique() # Randomly sample cases from the overrepresented label to match the underrepresented label
            selected_overrepresented_case_ids = np.random.choice(overrepresented_case_ids, size=num_cases_to_keep, replace=False)

            data = pd.concat([ 
                data[(data[self.label_col] == overrepresented_label) & (data[self.case_id_col] == case_id)]
                for case_id in selected_overrepresented_case_ids] 
                
                + [data[data[self.label_col] == the_other_label]] 
                + [data[data[self.label_col] == underrepresented_label]]) # Filter the DataFrame to include the sampled cases and all cases from the underrepresented label

        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=[self.case_id_col, self.label_col])
        label_counts = unique_cases[self.label_col].value_counts() # Count the labels again

        return data
    
    def undersample_prefixes(self, data):
        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])

        label_counts = unique_cases['label'].value_counts() # Calculate class distribution
        print('label counts before', label_counts)

        overrepresented_label = label_counts.idxmax() # Find the labels with their counts
        underrepresented_label = label_counts.idxmin()

        num_cases_to_keep = label_counts.min() # Count of cases to keep for each class (matching the underrepresented class)

        overrepresented_case_ids = data[data['label'] == overrepresented_label]['Case ID'].unique() # Randomly sample cases from the overrepresented label to match the underrepresented label
        selected_overrepresented_case_ids = np.random.choice(overrepresented_case_ids, size=num_cases_to_keep, replace=False)

        data = pd.concat([ 
            data[(data['label'] == overrepresented_label) & (data['Case ID'] == case_id)]
            for case_id in selected_overrepresented_case_ids
        ] + [
            data[data['label'] == underrepresented_label]
        ]) # Filter the DataFrame to include the sampled cases and all cases from the underrepresented label

        unique_cases = data.copy()
        unique_cases = unique_cases.drop_duplicates(subset=['Case ID', 'label'])
        label_counts = unique_cases['label'].value_counts() # Count the labels again
        print('label counts train after', label_counts)
        return data

    # https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2
    class ColumnEncoder(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.columns = None
            self.maps = dict()

        def transform(self, X):
            X_copy = X.copy()
            for col in self.columns:
                # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
                X_copy.loc[:, col] = X_copy.loc[:, col].apply(lambda x: self.maps[col].get(x, -1)+1)
            return X_copy
        
        def get_maps(self):
            return self.maps

        def inverse_transform(self, X):
            X_copy = X.copy()
            for col in self.columns:
                values = list(self.maps[col].keys())
                # find value in ordered list and map out of range values to None
                X_copy.loc[:, col] = [values[i-1] if 0 < i <= len(values) else None for i in X_copy[col]]
            return X_copy

        def fit(self, X, y=None):
            # only apply to string type columns
            self.columns = [col for col in X.columns if is_string_dtype(X[col])]
            for col in self.columns:
                self.maps[col] = OrderedDict({value: num for num, value in enumerate(sorted(set(X[col])))})
            return self
    
    def to_categorical_all(self, train, test, val, num_classes):
            """ 1-hot encodes a tensor """
            train_OHE = torch.tensor(np.eye(num_classes)[train])
            test_OHE = torch.tensor(np.eye(num_classes)[test])
            val_OHE = torch.tensor(np.eye(num_classes)[val])
            return train_OHE, test_OHE, val_OHE

    def vocabulary_size(self, data):
        no_cols_list = []
        for i in [self.activity_col]:
            _, _, _, no_cols = self.create_indexes(i, data)
            no_cols_list.append(no_cols)
        vocab_size = no_cols_list[0]+1 #the vocabulary size is one bigger because you will add the padding value 0
        return vocab_size