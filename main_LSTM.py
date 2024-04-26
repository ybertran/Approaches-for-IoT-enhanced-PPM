from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Input, LayerNormalization, BatchNormalization
from keras.utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Nadam
from sklearn import preprocessing
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from collections import Counter
import numpy as np
import copy
import csv
import time
from datetime import datetime
from math import log
import pandas as pd
pd.set_option('display.max_rows', None)
TF_ENABLE_ONEDNN_OPT = 0
from datetime import datetime
import time

###########################################################################################
# this loads and preprocesses the data. For now, we only use the activities and timefeature
###########################################################################################
iot = True
context = True
dummy = False
correlations = False

data_input_type = ''
if iot:
    data_input_type += '_iot'
if context:
    data_input_type += '_context'
if dummy:
    data_input_type += '_dummy'

"""context_event_labels = ['long fast mild downwards flow drift', 'long fast mild upwards flow drift',
                        'long fast strong downwards flow drift', 'long fast strong upwards flow drift',
                        'long middle mild downwards flow drift', 'long middle mild upwards flow drift',
                        'long middle strong downwards flow drift', 'long middle strong upwards flow drift',
                        'long middle weak downwards flow drift', 'long middle weak upwards flow drift',
                        'long slow mild downwards flow drift', 'long slow mild upwards flow drift',
                        'long slow strong downwards flow drift', 'long slow strong upwards flow drift',
                        'long slow weak downwards flow drift', 'long slow weak upwards flow drift'
                        'short fast mild downwards flow drift', 'short fast mild upwards flow drift',
                        'short middle mild downwards flow drift', 'short middle mild upwards flow drift',
                        'short middle weak downwards flow drift', 'short middle weak upwards flow drift',
                        'short slow mild downwards flow drift', 'short slow mild upwards flow drift',
                        'short slow weak downwards flow drift',  'short slow weak upwards flow drift']"""

filename = 'contextualised_full_context_process_event_data'

df = pd.read_csv('data/' + filename +'.csv') #read in original data
df.drop(columns= ['Unnamed: 0'], inplace=True)
df.loc[df['eventType'] == 'process', 'variable'] = 'process'
print(df.loc[df['eventType'] == 'process'])
# Drop rows with missing values
print(df.isna().value_counts())
df = df.dropna()
df = df[df['lifecycle:transition'].isin(['complete', 'ongoing'])]
df.rename(columns= {'timestamp': 'CompleteTimestamp'}, inplace=True)
df = df[~df['ActivityID'].isin(['pump stopped', 'pump starting'])]

"""if False: #context:
    # add context events
    context_df = pd.read_csv('C:/Users/u0139686/OneDrive - KU Leuven/IoT data model for PM/NICE model/Journal extension/drift_events.csv', index_col='Unnamed: 0')
    context_df.rename(columns={'timestamp': 'CompleteTimestamp', 'label': 'ActivityID', 'Batch ID': 'CaseID'}, inplace=True)
    context_df['lifecycle:transition'] = 'complete'
    context_df = context_df[context_df['ActivityID'].isin(context_event_labels)]

    df = pd.concat([df, context_df], axis= 0)
    df.sort_values(by='CompleteTimestamp', inplace=True)"""

if context:
    df.loc[df['eventType'] == 'context', 'ActivityID'] = df.loc[df['eventType'] == 'context', 'variable'] + ' ' + df.loc[df['eventType'] == 'context', 'ActivityID']
    print(df.head(50))

    #df = df[~ df['variable'].isin(['short term flow status', 'mid term flow status'])]

if not dummy:
    df = df[df['ActivityID'] != 'dummy'] # we only take the completed events
columns_to_keep = ['CaseID', 'ActivityID', 'CompleteTimestamp']

if iot:
    columns_to_keep += ['Filter 1 DeltaP_mean', 'Filter 1 DeltaP_ema', 'Filter 1 DeltaP_max', 'Filter 1 DeltaP_min', 'Filter 1 DeltaP_std',
                   'Filter 1 Inlet Pressure_mean', 'Filter 1 Inlet Pressure_ema', 'Filter 1 Inlet Pressure_max', 'Filter 1 Inlet Pressure_min', 'Filter 1 Inlet Pressure_std',
                   'Filter 2 DeltaP_mean', 'Filter 2 DeltaP_ema', 'Filter 2 DeltaP_max', 'Filter 2 DeltaP_min', 'Filter 2 DeltaP_std',
                   'Pump Circulation Flow_mean', 'Pump Circulation Flow_ema', 'Pump Circulation Flow_max', 'Pump Circulation Flow_min', 'Pump Circulation Flow_std',
                   'Tank Pressure_mean', 'Tank Pressure_ema', 'Tank Pressure_max', 'Tank Pressure_min', 'Tank Pressure_std']

df = df[columns_to_keep]
df = df.sort_values(by=['CaseID', 'CompleteTimestamp']) # Sort the DataFrame by 'CaseID' and 'CompleteTimestamp'
df = df.reset_index(drop=True) # If you want to reset the index after sorting and grouping

le_act = preprocessing.LabelEncoder() #labelencode the activity labels
le_case = preprocessing.LabelEncoder() #labelencode the case IDs

# Use .loc to avoid the SettingWithCopyWarning
activities = le_act.fit_transform(df['ActivityID'])
df.loc[:, 'CaseID'] = le_case.fit_transform(df['CaseID']) 
df = df.sort_values(['CaseID','CompleteTimestamp']) #sort values by CaseID
label_mapping = dict(zip(le_act.classes_, le_act.transform(le_act.classes_))) # Get the mapping of original labels to encoded values
print("Label Mapping:") # Display the label mapping
for original_label, encoded_label in label_mapping.items():
    print(f"{original_label} -> {encoded_label}")

df['ActivityID'] = activities

filename += data_input_type

filename += '_full_'
#if iot:
#    df.to_csv('data/' + filename + '_iot_reworked.csv', index=False)  # save and load again
#elif context:
#    df.to_csv('data/' + filename + '_context_reworked.csv', index=False)  # save and load again
#else:
df.to_csv('data/' + filename + 'reworked.csv', index=False) #save and load again
df = pd.read_csv('data/' + filename + 'reworked.csv')
print('the shape of the data', df.shape)

# these are just the variables we need to initialize

if iot:
    IoT_seqs = {col: [] for col in columns_to_keep if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # the IoT_seqs is a dictionary of column values(keys) and the values are the cases. Each case is a trace of length between 0 and 44
    IoT_lines = {col: [] for col in columns_to_keep if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # Create lists dynamically for each group of columns
lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4 = [], [], [], [], []
lastcase, line, firstLine = '', '', True
lines, times, times2, numlines, index = [], [], [], 0, 0
casestarttime, lasteventtime = None, None
step, sentences, softness, next_chars = 1, [], 0, [] # Vectorization
sentences_t, sentences_t2, sentences_t3, sentences_t4 = [], [], [], []
next_chars_t, next_chars_t2, next_chars_t3, next_chars_t4 = [], [], [], []
case_ids = []
#########################################################################################
#the next code snippet basically creates lists of lists with all the relevant information
#########################################################################################

eventlog = filename + 'reworked'
csvfile = open('data/%s' % eventlog+'.csv', 'r')
datareader = csv.reader(csvfile, delimiter=',', quotechar='|') # Load data from CSV file
next(datareader, None)  # skip the headers
for row in datareader: #the columns are "CaseID, ActivityID, CompleteTimestamp"
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S") # creates a datetime object from row[2]
    if row[0] != lastcase:  #'lastcase' is to save the last executed case for the loop. We only go in this if-statement if we have a new case
        casestarttime = t 
        lasteventtime = t
        lastcase = row[0]
        if not firstLine: # we always do this if-statement, only not for the first ever line (would add an empty list to a list)
            case_ids.append(row[0])
            lineseq.append(lines)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
            timeseqs4.append(times4)
            if iot:
                for index, (col, values) in enumerate(IoT_seqs.items()):
                    IoT_seqs[col].append(IoT_lines[col])
        if iot:
            IoT_lines = {col: [] for col in columns_to_keep if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']}
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
    if iot:
        for index, (col, values) in enumerate(IoT_lines.items()):
            IoT_lines[col].append(str(row[3 + index]))
    lasteventtime = t
    index = index+1
    firstLine = False #after the first line we set the FirstLine to False; this is because we want to save the lines and times lists to the lineseq and timeseq list for each seperate case

# add last case
case_ids.append(row[0])
lineseq.append(lines)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
timeseqs4.append(times4)
if iot:
    for index, (col, values) in enumerate(IoT_seqs.items()):
        IoT_seqs[col].append(IoT_lines[col])
numlines+=1

# Calculate divisors
divisor = np.mean([item for sublist in timeseqs for item in sublist]) # the average time difference between the event in a case and the previous event, across all events in the dataset
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist]) # average time difference since the start of a case across all events in the dataset.

print('divisor: {}'.format(divisor))
print('divisor2: {}'.format(divisor2))

maxlen = max(map(lambda x: len(x),lineseq))
chars = map(lambda x : set(x),lineseq)
chars = list(set().union(*chars))

target_chars = [1]

# Custom sorting function to convert strings to integers for numerical sorting
def custom_sort(item):
    return int(item)

# Sorting the list using the custom sorting function (because we sort strings based on their integer value)
chars = sorted(chars, key=custom_sort)

print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
print('maxlen', maxlen)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

#########################################################################################################
                        # the Niek Tax way of generating prefixes for the traces
#########################################################################################################

if iot:
    IoT_sentences = {col: [] for col in columns_to_keep if col not in ['CaseID', 'ActivityID', 'CompleteTimestamp']} # Initialize IoT sentences
    IoT_cols = list(IoT_seqs.keys())  # Get the keys of IoT_seqs

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
        if iot:
            for col in IoT_cols:
                IoT_sentences[col].append(IoT_seqs[col][counter][0:i]) # Now 'IoT_sentences' contains sequences for each column
        if line[i] == str(12):
            next_char = 1
        else:
            next_char = 0
        next_chars.append(next_char)

print('number of prefixes:', len(sentences)) # Print the number of prefixes

# From here on, we have the X and Y and add the information iteratively.

num_features = (len(chars) +5)
if iot:
    num_features+= len(IoT_sentences.keys()) # the number of activities and 5 time-based features
X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32) # OHE the input data, input of size [#prefixes, prefix_length, #features]
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
        if iot:
            for col in IoT_sentences.keys():
                X[i, t + leftpad, len(chars) + 5 + list(IoT_sentences.keys()).index(col)] = IoT_sentences[col][i][t]

    for c in target_chars:
        if c == next_chars[i]:
            y_a[i, target_char_indices[c]] = 1 # this is just equivalent to a standard one-hot encoding
        else:
            y_a[i, target_char_indices[c]] = 0

y_a = np.array(next_chars)

# sentences contains the prefixes
# y_a contains the target (next activity is pump adjustment or not)

if correlations:

    prefixes_df = pd.DataFrame(columns= ['caseid', 'activityid'])
    for i in range(len(sentences)):
        new_prefix = pd.DataFrame(columns= ['caseid', 'activityid'], index=range(len(sentences[i])))
        new_prefix['caseid'] = i
        new_prefix['activityid'] = sentences[i]

        prefixes_df = pd.concat([prefixes_df, new_prefix], axis=0)

    prefix_frequencies = prefixes_df.groupby(by= 'caseid').value_counts(subset= ['activityid'])
    prefix_frequencies= prefix_frequencies.unstack().fillna(0)

    prefix_frequencies['target'] = y_a
    print(prefix_frequencies)

    for column in prefix_frequencies.columns.drop('target'):
            print(column, ': ', prefix_frequencies[column].corr(other= prefix_frequencies['target']))

"""
sentences_df = pd.DataFrame(data=sentences)
prefixes_df = pd.DataFrame(columns=range(max(label_mapping.values())+1), index=range(len(sentences_df)))
print(sentences_df.value_counts())
for prefix in sentences_df.index:
    value = sentences_df.loc[prefix].value_counts()
    prefixes_df.loc[prefix] = value.transpose().fillna(0)
    #print(prefixes_df)
prefixes_df = prefixes_df.transpose()
print(prefixes_df.value_counts())

fdsqfdsgs
"""
#########################################################################################################
                        # Building the LSTM model. 
#########################################################################################################

# Build the model
print('Build model...')
main_input = Input(shape=(maxlen, num_features), name='main_input')

l1 = LSTM(100, implementation=2, return_sequences=True, dropout=0.2)(main_input)
b1 = LayerNormalization()(l1)
l2_1 = LSTM(100, implementation=2,  return_sequences=False, dropout=0.2)(b1)
b2_1 = LayerNormalization()(l2_1)

act_output = Dense(1, activation='sigmoid', name='act_output')(b2_1)

model = Model(inputs=[main_input], outputs=act_output)

opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
# Assuming y_train contains your class labels
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_a), y=y_a)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(class_weight_dict)

model.compile(loss={'act_output':'binary_crossentropy'}, optimizer=opt, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=42)

model_output_path = 'output_files/models/model_{epoch:02d}-{val_loss:.2f}_'+ str(X.shape) + data_input_type + '_full_status'
#if iot:
#    model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}_'+ str(X.shape)+'IoT_cols'+str(len(IoT_cols))+'.keras', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
#elif context:
#    model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}_'+ str(X.shape)+'context' + '.keras', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
#else:
#    model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}_'+ str(X.shape) + '.keras', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

model_checkpoint = ModelCheckpoint(model_output_path + '.keras', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model.fit(X, y_a, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], class_weight=class_weight_dict, batch_size=maxlen, epochs=100)