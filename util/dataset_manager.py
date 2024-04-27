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
from sklearn import preprocessing
import csv
import time
from datetime import datetime

class DatasetManager:

    def __init__(self, dataset_name, context, dummy, iot):
        self.dataset_name = dataset_name
        self.case_id_col = 'CaseID'
        self.activity_col = 'ActivityID'
        self.timestamp_col = 'CompleteTimestamp'
        self.context = context
        self.dummy = dummy
        self.iot = iot
    
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
    
    def read_data(self):
        df = pd.read_csv('data/' + self.dataset_name +'.csv') 
        return df
        
    def preprocess_data(self, df, data_input_type):
        """
        Preprocesses the given DataFrame by performing various data cleaning and transformation operations.

        Args:
            df (pandas.DataFrame): The input DataFrame to be preprocessed.
            data_input_type (str): The type of data input.

        Returns:
            pandas.DataFrame: The preprocessed DataFrame.
            str: The filename of the preprocessed data.

        """
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.loc[df['eventType'] == 'process', 'variable'] = 'process'
        df = df.dropna()
        df = df[df['lifecycle:transition'].isin(['complete', 'ongoing'])]
        df.rename(columns={'timestamp': 'CompleteTimestamp'}, inplace=True)
        df = df[~df['ActivityID'].isin(['pump stopped', 'pump starting'])]

        if self.context:
            df.loc[df['eventType'] == 'context', 'ActivityID'] = df.loc[df['eventType'] == 'context', 'variable'] + ' ' + df.loc[df['eventType'] == 'context', 'ActivityID']

        if not self.dummy:
            df = df[df['ActivityID'] != 'dummy']

        columns_to_keep = ['CaseID', 'ActivityID', 'CompleteTimestamp']

        if self.iot:
            columns_to_keep += ['Filter 1 DeltaP_mean', 'Filter 1 DeltaP_ema', 'Filter 1 DeltaP_max', 'Filter 1 DeltaP_min', 'Filter 1 DeltaP_std',
                                'Filter 1 Inlet Pressure_mean', 'Filter 1 Inlet Pressure_ema', 'Filter 1 Inlet Pressure_max', 'Filter 1 Inlet Pressure_min', 'Filter 1 Inlet Pressure_std',
                                'Filter 2 DeltaP_mean', 'Filter 2 DeltaP_ema', 'Filter 2 DeltaP_max', 'Filter 2 DeltaP_min', 'Filter 2 DeltaP_std',
                                'Pump Circulation Flow_mean', 'Pump Circulation Flow_ema', 'Pump Circulation Flow_max', 'Pump Circulation Flow_min', 'Pump Circulation Flow_std',
                                'Tank Pressure_mean', 'Tank Pressure_ema', 'Tank Pressure_max', 'Tank Pressure_min', 'Tank Pressure_std']

        df = df[columns_to_keep]

        self.columns = columns_to_keep
        df = df.sort_values(by=['CaseID', 'CompleteTimestamp'])
        df = df.reset_index(drop=True)

        le_act = preprocessing.LabelEncoder()
        le_case = preprocessing.LabelEncoder()

        activities = le_act.fit_transform(df['ActivityID'])
        df.loc[:, 'CaseID'] = le_case.fit_transform(df['CaseID'])
        df = df.sort_values(['CaseID', 'CompleteTimestamp'])
        label_mapping = dict(zip(le_act.classes_, le_act.transform(le_act.classes_)))
        for original_label, encoded_label in label_mapping.items():
            print(f"{original_label} -> {encoded_label}")

        df['ActivityID'] = activities

        filename = self.dataset_name
        filename += data_input_type

        df.to_csv('data/' + filename + 'reworked.csv', index=False)
        df = pd.read_csv('data/' + filename + 'reworked.csv')
        print('the shape of the data', df.shape)

        return df, filename
    
    # Custom sorting function to convert strings to integers for numerical sorting
    def custom_sort(self,item):
        return int(item)

    def create_relevant_data(self, eventlog):
        """
        Creates relevant data from the given eventlog.

        Args:
            eventlog (str): The name of the eventlog.

        Returns:
            tuple: A tuple containing the following:
                - lineseq (list): A list of activity labels for each case.
                - timeseqs (list): A list of time differences between consecutive events for each case.
                - timeseqs2 (list): A list of time differences between the start of each case and each event.
                - timeseqs3 (list): A list of time differences between each event and midnight.
                - timeseqs4 (list): A list of day of the week for each event.
                - IoT_seqs (dict): A dictionary containing IoT data for each case, if `iot` is True.
        """
        lastcase, firstLine = '', True
        lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4 = [], [], [], [], []
        lines, times, times2, numlines, index = [], [], [], 0, 0
        casestarttime, lasteventtime = None, None

        csvfile = open('data/%s' % eventlog+'.csv', 'r')
        datareader = csv.reader(csvfile, delimiter=',', quotechar='|') # Load data from CSV file
        next(datareader, None)  # skip the headers
        
        if self.iot: 
            IoT_seqs = {col: [] for col in self.columns if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # the IoT_seqs is a dictionary of column values(keys) and the values are the cases. Each case is a trace of length between 0 and 44
            IoT_lines = {col: [] for col in self.columns if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # Create lists dynamically for each group of columns

        lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4 = [], [], [], [], []
        for row in datareader: #the columns are "CaseID, ActivityID, CompleteTimestamp"
            t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S") # creates a datetime object from row[2]
            if row[0] != lastcase:  #'lastcase' is to save the last executed case for the loop. We only go in this if-statement if we have a new case
                casestarttime = t 
                lasteventtime = t
                lastcase = row[0]
                if not firstLine: # we always do this if-statement, only not for the first ever line (would add an empty list to a list)
                    lineseq.append(lines)
                    timeseqs.append(times)
                    timeseqs2.append(times2)
                    timeseqs3.append(times3)
                    timeseqs4.append(times4)
                    if self.iot:
                        for index, (col, values) in enumerate(IoT_seqs.items()):
                            IoT_seqs[col].append(IoT_lines[col])
                if self.iot:
                    IoT_lines = {col: [] for col in self.columns if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']}
                lines = []
                times = []
                times2 = []
                times3 = []
                times4 = []
                numlines+=1

            timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
            timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
            timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds     #multiply with 60*60*24 = 86400 to go from days to seconds
            timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds    #the .seconds method gives the time in seconds
            timediff3 = timesincemidnight.seconds #this leaves only time event occured after midnight
            timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday() #day of the week
            lines.append(str(row[1])) #add the activity label to the line list
            times.append(timediff)
            times2.append(timediff2)
            times3.append(timediff3)
            times4.append(timediff4)
            if self.iot:
                for index, (col, values) in enumerate(IoT_lines.items()):
                    IoT_lines[col].append(str(row[3 + index]))
            lasteventtime = t
            firstLine = False #after the first line we set the FirstLine to False; this is because we want to save the lines and times lists to the lineseq and timeseq list for each seperate case

        # add last case
        lineseq.append(lines)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
        
        if self.iot:
            for index, (col, values) in enumerate(IoT_seqs.items()):
                IoT_seqs[col].append(IoT_lines[col])
        numlines+=1
        
        return lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4, IoT_seqs
    
    def create_index(self, lineseq, timeseqs, timeseqs2, IoT_seqs):
        """
        Creates an index for the given dataset.

        Args:
            lineseq (list): A list of sequences representing the lines in the dataset.
            timeseqs (list): A list of sequences representing the time differences between events in a case.
            timeseqs2 (list): A list of sequences representing the time differences since the start of a case.
            IoT_seqs (dict): A dictionary containing IoT sequences.
            iot (bool): A flag indicating whether IoT sequences are included.

        Returns:
            tuple: A tuple containing various index-related objects:
                - chars (list): A list of unique characters found in the dataset.
                - maxlen (int): The maximum length of a line sequence.
                - divisor (float): The average time difference between events in a case.
                - divisor2 (float): The average time difference since the start of a case.
                - char_indices (dict): A dictionary mapping characters to their indices.
                - indices_char (dict): A dictionary mapping indices to characters.
                - target_char_indices (dict): A dictionary mapping target characters to their indices.
                - target_indices_char (dict): A dictionary mapping indices to target characters.
                - target_chars (list): A list of target characters.
                - IoT_cols (list): A list of IoT column names.
                - IoT_sentences (dict): A dictionary mapping IoT column names to empty lists.
        """
        # Calculate divisors
        divisor = np.mean([item for sublist in timeseqs for item in sublist]) # the average time difference between the event in a case and the previous event, across all events in the dataset
        divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) # average time difference since the start of a case across all events in the dataset.

        print('divisor: {}'.format(divisor))
        print('divisor2: {}'.format(divisor2))

        maxlen = max(map(lambda x: len(x),lineseq))
        chars = map(lambda x : set(x),lineseq)
        chars = list(set().union(*chars))

        target_chars = [1]

        # Sorting the list using the custom sorting function (because we sort strings based on their integer value)
        chars = sorted(chars, key=self.custom_sort)

        print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
        print('maxlen', maxlen)
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
        target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

        if self.iot:
            IoT_sentences = {col: [] for col in self.columns if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # Initialize IoT sentences
            IoT_cols = list(IoT_seqs.keys())  # Get the keys of IoT_seqs
            
        self.num_features = (len(chars) + 5) # the number of activities and 5 time-based features
        return chars, maxlen, divisor, divisor2, char_indices, indices_char, target_char_indices, target_indices_char, target_chars, IoT_cols, IoT_sentences
    
    
    def create_OHE_data(self, lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4, IoT_seqs, chars, maxlen, divisor, divisor2, char_indices, target_char_indices, target_chars, IoT_cols, IoT_sentences):
        
        """These are the variables that we need to initialize
        """
        step, sentences, next_chars = 1, [], [] # Vectorization
        sentences_t, sentences_t2, sentences_t3, sentences_t4 = [], [], [], []

                        
        #########################################################################################################
                                # the Niek Tax way of generating prefixes for the traces
        #########################################################################################################

        # In this for loop, we generate the lists sentences, time_sentences and IOT sentences. These contain the information that will be added to the X and Y arrays.
        for counter, (line, line_t, line_t2, line_t3, line_t4) in enumerate(zip(lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4)):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # Append sequences for 'lineseq', 'timeseqs', and IoT sequences
                sentences.append(line[0:i])
                sentences_t.append(line_t[0:i])
                sentences_t2.append(line_t2[0:i])
                sentences_t3.append(line_t3[0:i])
                sentences_t4.append(line_t4[0:i])
                # Append IoT sequences
                if self.iot:
                    for col in IoT_cols:
                        IoT_sentences[col].append(IoT_seqs[col][counter][0:i]) # Now 'IoT_sentences' contains sequences for each column
                if line[i] == str(12):
                    next_char = 1
                else:
                    next_char = 0
                next_chars.append(next_char)

        print('number of prefixes:', len(sentences)) # Print the number of prefixes

        # From here on, we have the X and Y and add the information iteratively.

        if self.iot:
            self.num_features+= len(IoT_sentences.keys()) # the number of activities and 5 time-based features
        X = np.zeros((len(sentences), maxlen, self.num_features), dtype=np.float32) # OHE the input data, input of size [#prefixes, prefix_length, #features]
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32) # With categorical_crossentropy, your output_files does not need to be OHE

        """
        The input is of size [#prefixes, prefix length, #features]. 
        The first 15 features are the activity values, the next 5 features are time-based features

        - the event number
        - the normalized timesincelastevent between the event and the previous event
        - the normalized timesincecasestart between the event and the start of the case
        - the normalized timesincemidnight between the event and the midnight time
        - the normalized week of the day feature
        """
        for i, sentence in enumerate(sentences):
            leftpad = maxlen-len(sentence) # calculates the padding that we need to do; we do leftpadding
            sentence_t = sentences_t[i] # the next time feature values
            sentence_t2 = sentences_t2[i]
            sentence_t3 = sentences_t3[i]
            sentence_t4 = sentences_t4[i]

            for t, char in enumerate(sentence): # to run over the prefix
                for c in chars:
                    if c==char: #this will encode present events to the right places
                        X[i, t+leftpad, char_indices[c]] = 1

                X[i, t+leftpad, len(chars)] = t+1
                X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
                X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
                X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/86400
                X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7

                # Add information for IoT columns
                if self.iot:
                    for col in IoT_sentences.keys():
                        X[i, t + leftpad, len(chars) + 5 + list(IoT_sentences.keys()).index(col)] = IoT_sentences[col][i][t]

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 # this is just equivalent to a standard one-hot encoding
                else:
                    y_a[i, target_char_indices[c]] = 0

        y_a = np.array(next_chars)
        
        return X, y_a, sentences