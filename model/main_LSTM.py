import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Add the parent directory of the current script to the Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import LSTM, Dense, Input, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from util.dataset_manager import DatasetManager

pd.set_option('display.max_rows', None)

###################################################################################################################
############################## The main LSTM file #################################################################
###################################################################################################################

"""
In this file, we read in the data, preprocess it and train the model. The data is read in from a CSV file, which is then preprocessed."""
iot = True
context = True
dummy = False
correlations = False

# Define the data input type
data_input_type = ''
if iot:
    data_input_type += '_iot'
if context:
    data_input_type += '_context'
if dummy:
    data_input_type += '_dummy'

# The context event labels (in total there are 24 context events)
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
datasetmanager = DatasetManager(filename, context, dummy, iot)
df = datasetmanager.read_data() # Read in the data
df, filename = datasetmanager.preprocess_data(df, data_input_type) # Preprocess the data

#########################################################################################
#the next code snippet basically creates lists of lists with all the relevant information
#########################################################################################

eventlog = filename + 'reworked'
# this function creates the 
lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4, IoT_seqs = datasetmanager.create_relevant_data(eventlog)
chars, maxlen, divisor, divisor2, char_indices, indices_char, target_char_indices, target_indices_char, target_chars, IoT_cols, IoT_sentences = datasetmanager.create_index(lineseq, timeseqs, timeseqs2, IoT_seqs)
X, y_a, sentences = datasetmanager.create_OHE_data(lineseq, timeseqs, timeseqs2, timeseqs3, timeseqs4, IoT_seqs, chars, maxlen, divisor, divisor2, char_indices, target_char_indices, target_chars, IoT_cols, IoT_sentences)

# X contains the prefixes
# sentences contains the sentences
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

#########################################################################################################
                        # Building the LSTM model. 
#########################################################################################################

# Build the model
print('Build model...')
main_input = Input(shape=(maxlen, X.shape[2]), name='main_input')

l1 = LSTM(100, implementation=2, return_sequences=True, dropout=0.2)(main_input)
b1 = LayerNormalization()(l1)
l2_1 = LSTM(100, implementation=2,  return_sequences=False, dropout=0.2)(b1)
b2_1 = LayerNormalization()(l2_1)

act_output = Dense(1, activation='sigmoid', name='act_output')(b2_1)

model = Model(inputs=[main_input], outputs=act_output)

opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=3)
# Assuming y_train contains your class labels

class_weights = compute_class_weight('balanced', classes=np.unique(y_a), y=y_a)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print(class_weight_dict)

model.compile(loss={'act_output':'binary_crossentropy'}, optimizer=opt, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_output_path = 'output_files/models/model_{epoch:02d}-{val_loss:.2f}_'+ str(X.shape) + data_input_type + '_full_status'
model_checkpoint = ModelCheckpoint(model_output_path + '.keras', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

model.fit(X, y_a, validation_split=0.2, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], class_weight=class_weight_dict, batch_size=maxlen, epochs=100)