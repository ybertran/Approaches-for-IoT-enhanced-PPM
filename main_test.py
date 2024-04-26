###### IMPORT FUNCTIONS
from __future__ import division, print_function
import pandas as pd

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
df = pd.read_csv('data/event_data_aggregated_alex.csv', index_col ="Unnamed: 0")

print(list(df.columns))

df2 = df[df['lifecycle:transition']=='complete']

columns_to_keep = ['CaseID', 'CompleteTimestamp', 'Vessel', 'ActivityID', 'lifecycle:transition', 'event_id', 'activityID', 
                   'Filter 1 DeltaP_mean', 'Filter 1 DeltaP_ema', 'Filter 1 DeltaP_max', 'Filter 1 DeltaP_min', 'Filter 1 DeltaP_std', 
                   'Filter 1 Inlet Pressure_mean', 'Filter 1 Inlet Pressure_ema', 'Filter 1 Inlet Pressure_max', 'Filter 1 Inlet Pressure_min', 'Filter 1 Inlet Pressure_std', 
                   'Filter 2 DeltaP_mean', 'Filter 2 DeltaP_ema', 'Filter 2 DeltaP_max', 'Filter 2 DeltaP_min', 'Filter 2 DeltaP_std', 
                   'Pump Circulation Flow_mean', 'Pump Circulation Flow_ema', 'Pump Circulation Flow_max', 'Pump Circulation Flow_min', 'Pump Circulation Flow_std', 
                   'Tank Pressure_mean', 'Tank Pressure_ema', 'Tank Pressure_max', 'Tank Pressure_min', 'Tank Pressure_std']

df2 = df2[columns_to_keep]

# Sort the DataFrame by 'CaseID' and 'CompleteTimestamp'
df_sorted = df2.sort_values(by=['CaseID', 'CompleteTimestamp'])

# Create a new column 'event_nr' using groupby and cumcount
df2['event_nr'] = df2.groupby('CaseID').cumcount() + 1

# If you want to reset the index after sorting and grouping
df2 = df2.reset_index(drop=True)

#df2.drop(columns=['CompleteTimestamp'])

print(df2.head())

df2.to_csv('data/event_data_aggregated_alex2.csv')