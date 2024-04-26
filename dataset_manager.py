import os
import util.dataset_confs as dataset_confs
import util.EncoderFactory as EncoderFactory
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from collections import OrderedDict
from pandas.api.types import is_string_dtype
import h5py
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:

    def __init__(self, dataset_name, case_id_col, activity_col, timestamp_col):
        self.dataset_name = dataset_name
        self.case_id_col = 'CaseID'
        self.activity_col = 'ActivityID'
        self.timestamp_col = 'CompleteTimestamp'

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        case_length = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        data.loc[:, 'case_length'] = case_length.copy()
        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s" % (x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))

        return dt_prefixes