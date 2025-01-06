import matplotlib.pyplot as plt
import pandas as pd
#from PIL.ImageOps import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import time
import json
from pandas import MultiIndex
from xgboost import XGBClassifier

from util.iot_data_loader import DataLoader,TimeSeriesDataLoader
import os.path as path
from os import mkdir
import numpy as np

##########################################
# define paths, variables and parameters #
##########################################

repository_path = 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/data/'
targets_path = 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/process_event_data2.csv'
save_results = False
dl = DataLoader(path='repository_path')

event_data = pd.read_csv(targets_path, usecols=['CaseID', 'CompleteTimestamp', 'Vessel', 'ActivityID', 'lifecycle:transition'], parse_dates=['CompleteTimestamp'])
event_data = event_data.loc[event_data['lifecycle:transition'] == 'complete']
print(len(event_data.loc[event_data['ActivityID'] == 'Pump adjustment']))
batches_list = event_data['CaseID'].unique()
iot_parameters = ['Pump Circulation Flow', 'Filter 1 DeltaP', 'Filter 1 Inlet Pressure', 'Filter 2 DeltaP', 'Tank Pressure']

# define data hyperparameters
iot_granularities = ['1h', '15min', '5min', '1min'] # granularity level at which IoT data are aggregated
lag_windows = [0, 5, 10, 25, 50, 100] # number of lags to add to the input data (= number of time steps model can see in the past)
data_split = {'train': 0.7, 'val': 0.2, 'test': 0.1}
target = 'pump adjustment is next activity' # how the target variable is defined

# define model hyperparameters
max_depth = [2, 5, 10]
learning_rate = [0.01, 0.1, 0.3]
n_estimators = [20, 50, 100, 200]
subsample = [0.5, 0.8, 1.0]
min_child_weight = [2, 5, 10]

###################
# preprocess data #
###################

# loop over parameters
for iot_granularity in iot_granularities:
    print(f'Current IoT granularity = {iot_granularity}')
    if target == 'pump adjustment present in next iot granularity':
        time_horizon = [iot_granularity]

    for lag_window in lag_windows:

        print(f'Current lag window: {lag_window}')

        X = pd.DataFrame()
        y = pd.DataFrame()

        # loop over batches

        for batch in batches_list:
            file_path = path.join(repository_path, 'Sensor data ' + batch + '.csv')
            iot_data = pd.read_csv(file_path, sep=';', index_col='New Timestamp', parse_dates=True)

            batch_event_data = event_data.loc[event_data['CaseID'] == batch]
            batch_event_data.reset_index(inplace=True, drop=True)

            # define target following rule: if next event is pump adjustment, target == 1
            X_y_index = MultiIndex.from_arrays(arrays=[np.full(len(batch_event_data),batch),batch_event_data.index], names=['CaseID', 'New Timestamp'])
            y_batch = pd.Series(index=X_y_index)
            X_batch = pd.DataFrame(index=X_y_index, columns= [col + '_lag_' + str(lag) for col in iot_parameters for lag in range(lag_window+1)], dtype = 'float64')

            # loop over events to check rules and populate the target series
            for event in batch_event_data.index:

                #########################################
                # construct target variable accordingly #
                #########################################

                try:
                    next_event = batch_event_data.at[event+1, 'ActivityID']
                    if 'Pump adjustment' == next_event:
                        y_batch.loc[(batch, event)] = 1
                    else:
                        y_batch.loc[(batch, event)] = 0

                except KeyError:
                    break

                ##################################################################
                # aggregate IoT data to chosen granularity and lag window length #
                ##################################################################

                event_timestamp = batch_event_data.loc[event,'CompleteTimestamp']
                for parameter in iot_parameters:
                    for lag in range(lag_window+1):
                        iot_data_lag = iot_data.loc[event_timestamp - lag*pd.Timedelta(iot_granularity): event_timestamp - (lag-1)*pd.Timedelta(iot_granularity), parameter].mean()
                        X_batch.loc[(batch, event), parameter + '_lag_' + str(lag)] = iot_data_lag

            X_batch.dropna(inplace=True)
            y_batch.dropna(inplace=True)

            # concatenate batches to obtain one data and one target dataframes
            X = pd.concat([X, X_batch], axis=0)
            y = pd.concat([y,y_batch], axis=0)

        print(len(event_data.loc[event_data['ActivityID'] == 'Pump adjustment']))
        print(y.sum())

        X, y = X.loc[X.index.intersection(y.index)], y.loc[X.index.intersection(y.index)]
        print(X.shape)

        ##########################################
        # training and testing datasets creation #
        ##########################################

        # split dataset in training, validation and test sets (following the data_split dictionary)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=data_split['val'] + data_split['test'], random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=data_split['test']/(data_split['val'] + data_split['test']), random_state=42)

        if save_results:
            X_test.to_csv('generated datasets/X_test ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows.csv', index=True)
            y_test.to_csv('generated datasets/y_test ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows.csv', index=True)

        # handle class imbalance
        negative_count = (y_train == 0).sum()[0]
        positive_count = (y_train == 1).sum()[0]
        scale_pos_weight = negative_count / positive_count
        print(f"ratio of negative counts: {negative_count}/{positive_count} = {scale_pos_weight}")

        ##################
        # Model training #
        ##################

        start_time = time.time()
        print(f'Start of training: {time.ctime(start_time)}')

        # create the model
        xgb_model = XGBClassifier(scale_pos_weight = scale_pos_weight)#eval_metric='log_loss')

        # set parameters for the XGBoost model

        param_grid = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'min_child_weight': min_child_weight
        }

        # perform grid search
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                                   scoring='accuracy', cv=3, verbose=0, n_jobs=-1)

        # fit the grid search to the training data

        grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

        end_time = time.time()
        training_time = {'training time': end_time - start_time}
        print(f'Training completed after {training_time['training time']} seconds')

        #######################
        # Evaluate best model #
        #######################

        # Get the best model
        best_model = grid_search.best_estimator_
        hyperparameters = best_model.get_params()

        # print the best parameters and best score
        print(f"Best parameters for granularity {iot_granularity} and window length {lag_window}:", hyperparameters)
        print("Best cross-validation accuracy:", grid_search.best_score_)

        # Make predictions on the test set
        y_pred = best_model.predict(X_test)

        # Generate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        # Optional: Print a detailed classification report
        print("Classification Report:\n", classification_report(y_test, y_pred))

        ###################################
        # Save best model and xgb results #
        ###################################

        if save_results:
            # create directory to save model and xgb results together
            save_path = f'xgb results/model {data_split['train']} - {data_split['val']} - {data_split['test']}/'
            if not path.exists(save_path):
                mkdir(save_path)

            save_path += iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows ' + target + ' target output prob/'
            if not path.exists(save_path):
                mkdir(save_path)

            # save model
            best_model.save_model(save_path + "model.json")

            # save model hyperparameters
            with open(save_path + "xgb_parameters.json", "w") as f:
                json.dump(hyperparameters, f)

            with open(save_path + 'training time', 'w') as f:
                json.dump(training_time, f)

            # save confusion matrix
            cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(cm.shape[0])],
                                 columns=[f"Pred_{i}" for i in range(cm.shape[1])])
            cm_df.to_csv(save_path + "confusion_matrix_with_report.csv", index=True)

            ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            plt.savefig(save_path + 'confusion matrix.png')
            plt.close()