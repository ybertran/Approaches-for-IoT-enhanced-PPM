import matplotlib.pyplot as plt
import pandas as pd
from PIL.ImageOps import scale
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
targets_path = 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/process_event_data.csv'

dl = DataLoader(path='repository_path')

event_data = pd.read_csv(targets_path, usecols=['CaseID', 'CompleteTimestamp', 'Vessel', 'ActivityID', 'lifecycle:transition'], parse_dates=['CompleteTimestamp'])
batches_list = event_data['CaseID'].unique()

# define data hyperparameters
iot_granularities = ['1min'] #['1h', '15min', '5min', '1min'] # granularity level at which IoT data are aggregated
lag_windows = [5, 10, 25, 50, 100] # number of lags to add to the input data (= number of time steps model can see in the past)
time_horizon = ['15min', '30min', '1h', '2h'] # time window in which next pump adjustment should occur
#target = ['pump adjustment present in next time window', 'pump adjustment is next activity']

# define model hyperparameters
max_depth = [2, 5, 10]#, 15, 20],
learning_rate = [0.01, 0.1, 0.3]
n_estimators = [20, 50, 100, 200]
subsample = [0.5, 0.8, 1.0]
min_child_weight = [2, 5, 10]#, 15, 20]

###################
# preprocess data #
###################

# loop over parameters
for iot_granularity in iot_granularities:
    print(f'Current IoT granularity = {iot_granularity}')

    for lag_window in lag_windows:
        print(f'Current lag window: {lag_window}')

        for horizon in time_horizon:
            if pd.Timedelta(horizon) < pd.Timedelta(iot_granularity):
                continue

            print(f'Current time horizon: {horizon}')

            X = pd.DataFrame()
            y = pd.DataFrame()

            # loop over batches

            for batch in batches_list:
                file_path = path.join(repository_path, 'Sensor data ' + batch + '.csv')
                iot_data = pd.read_csv(file_path, sep=';', index_col='New Timestamp', parse_dates=True)

                #########################################
                # aggregate to chosen granularity and lag window length
                #########################################

                iot_data_agg = iot_data.resample(iot_granularity).mean()
                iot_data_lag = dl.add_lags(iot_data_agg, 'index', ['Filter 1 DeltaP', 'Filter 1 Inlet Pressure', 'Filter 2 DeltaP', 'Pump Circulation Flow', 'Tank Pressure'], num_lags=lag_window)
                iot_data_lag.ffill(inplace=True)
                iot_data_lag.dropna(inplace=True)

                #########################################
                # construct target variable accordingly #
                #########################################

                batch_event_data = event_data.loc[event_data['CaseID'] == batch]
                batch_event_data.set_index('CompleteTimestamp', inplace=True)

                # define target following rule A: if next event is pump adjustment, target == 1
                #y_next = pd.Series(index=iot_data_lag.index, data=np.zeros(len(iot_data_lag)))

                # define target following rule B: if next event present in coming time window, target == 1
                y_index = MultiIndex.from_arrays(arrays=[np.full(len(iot_data_lag),batch),iot_data_lag.index], names=['CaseID', 'New Timestamp'])
                y_pres = pd.Series(index=y_index, data=np.zeros(len(iot_data_lag)))

                # loop over time index to check rules and populate the target series
                for window in iot_data_lag.index:

                    # rule A
                    #next_event = list(batch_event_data.loc[window:,'ActivityID'])[0]
                    #if 'Pump adjustment' == next_event:
                    #    y_next.loc[window] = 1

                    # rule B
                    window_end = window + pd.Timedelta(horizon)
                    next_window_events = list(batch_event_data.loc[window:window_end, 'ActivityID'])
                    if 'Pump adjustment' in next_window_events:
                        y_pres.loc[(batch,window)] = 1

                iot_data_lag['CaseID'] = batch

                # concatenate batches to obtain one data and one target dataframes

                X = pd.concat([X, iot_data_lag], axis=0)
                y = pd.concat([y,y_pres], axis=0)

            print(len(event_data.loc[event_data['ActivityID'] == 'Pump adjustment']))
            print(y.sum())
            y.sort_index(inplace=True)

            # slightly reformat X dataframe by defining the case ID as an index for easier access
            X.reset_index(inplace=True)
            X.set_index(['CaseID', 'New Timestamp'], inplace=True)
            X.sort_index(inplace=True)

            ##########################################
            # training and testing datasets creation #
            ##########################################

            # split dataset in training and test sets (80-20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # handle class imbalance
            negative_count = (y_train == 0).sum()[0]
            positive_count = (y_train == 1).sum()[0]
            scale_pos_weight = negative_count / positive_count
            print(f"ratio of negative counts: {negative_count}/{positive_count} = {scale_pos_weight}")

            ##################
            # Model training #
            ##################

            start_time = time.time()

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
            grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

            end_time = time.time()
            training_time = {'training time': end_time - start_time}

            #######################
            # Evaluate best model #
            #######################

            # print the best parameters and best score
            print(f"Best parameters for granularity {iot_granularity} and window length {lag_window}:", grid_search.best_params_)
            print("Best cross-validation accuracy:", grid_search.best_score_)

            # Get the best model
            best_model = grid_search.best_estimator_

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

            # create directory to save model and xgb results together
            save_path = 'xgb results/model ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows ' + horizon + ' time horizon/'
            if not path.exists(save_path):
                mkdir(save_path)

            # save model
            best_model.save_model(save_path + "model.json")
            hyperparameters = best_model.get_params()

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