from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, f1_score, precision_score,
                             recall_score)
import pandas as pd
from os import path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

iot_granularities = ['15min'] # granularity level at which IoT data are aggregated
lag_windows = [25] # number of lags to add to the input data (= number of time steps model can see in the past)
targets = ['pump adjustment is next activity']
directory_paths = ['xgb results/', 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/xgb results/']
data_split = {'train': 0.7, 'val': 0.2, 'test': 0.1}

def visualise_f1_heatmap(target, lag_window, iot_granularities, time_horizon):
    data = {
        "iot_granularity": [],
        "horizon": [],
        "f1_score": []
    }

    for iot_granularity in iot_granularities:

        for horizon in time_horizon:
            if pd.Timedelta(horizon) >= pd.Timedelta(iot_granularity):
                if target == 'pump adjustment present in next time window':
                    file_path = 'xgb results/model ' + iot_granularity + ' iot granularity - ' + str(
                        lag_window) + ' lag windows ' + horizon + ' time horizon/confusion_matrix_with_report.csv'
                elif target == 'pump adjustment is next activity':
                    file_path = 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/xgb results/model ' + iot_granularity + ' iot granularity - ' + str(
                        lag_window) + ' lag windows 24h time horizon ' + target + ' target/confusion_matrix_with_report.csv'
                # file_path = directory_path + 'model ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows ' + horizon + ' time horizon/confusion_matrix_with_report.csv'
                if path.exists(file_path):
                    cm_df = pd.read_csv(file_path, index_col='Unnamed: 0')
                    print(cm_df)
                    tp = cm_df.loc['True_1', 'Pred_1']
                    fp = cm_df.loc['True_0', 'Pred_1']
                    fn = cm_df.loc['True_1', 'Pred_0']
                    f1_score = (2 * tp) / (2 * tp + fp + fn)
                    data['iot_granularity'].append(iot_granularity)
                    data['horizon'].append(horizon)
                    data['f1_score'].append(f1_score)
                    print(
                        f'F1 score for model with {iot_granularity} iot granularity, {str(lag_window)} lag windows, {horizon} time horizon and {target} target: {f1_score}')

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Pivot the DataFrame to create a matrix
    heatmap_data = df.pivot(index="iot_granularity", columns="horizon", values="f1_score")

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score'})
    plt.title("F1 Score Heatmap")
    plt.xlabel("Horizon")
    plt.ylabel("IoT granularity")
    plt.show()

def evaluate_auc(iot_granularities, lag_windows, target='pump adjustment is next activity'):

    model_directory = f'xgb results/model {data_split['train']} - {data_split['val']} - {data_split['test']}/'

    for iot_granularity in iot_granularities:
        print(f'Current IoT granularity = {iot_granularity}')

        for lag_window in lag_windows:
            print(f'Current lag window: {lag_window}')

            X_test = pd.read_csv('generated datasets/X_test ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows.csv', index_col=['CaseID', 'New Timestamp'])
            y_test = pd.read_csv('generated datasets/y_test ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows.csv', index_col='Unnamed: 0')

            # reindexing necessary to sort the X and y dataframes similarly and make them correspond with event log to obtain time before adjustments later
            y_test.index = [tuple(i.strip("()").split(", ")) for i in y_test.index]
            y_test.index = pd.MultiIndex.from_tuples(y_test.index, names=['BatchID', 'EventID'])
            y_test.index = pd.MultiIndex.from_arrays(
                [y_test.index.get_level_values('BatchID'), y_test.index.get_level_values('EventID').astype(int)],
                names=['BatchID', 'EventID']
            )

            X_test.sort_index(inplace=True)
            y_test.sort_index(inplace=True)

            # load the model
            model_path = model_directory + iot_granularity + ' iot granularity - ' + str(
                lag_window) + ' lag windows ' + target + ' target/'

            if path.exists(model_path):
                loaded_model = xgb.XGBClassifier()

                loaded_model.load_model(model_path + 'model.json')
                loaded_model.n_classes_ = 2

                # use the model for prediction on the test set
                y_pred = loaded_model.predict(X_test)
                y_prob = loaded_model.predict_proba(X_test)

                y_prob1 = y_prob[:,1]

                # store all the results in a dataframe and save it
                results = pd.DataFrame(columns= ['label', 'prediction', 'probabilities'])
                results['label'] = y_test
                results['prediction'] = y_pred
                results['probabilities'] = y_prob1 #, columns=['label', 'prediction', 'probabilities'])
                results.to_csv('C:/Users/yabertra/OneDrive - UGent/IoT PPM/test results/xgb 15 min 25 lags results.csv')

                # print various quality metrics
                print(f'AuC: {roc_auc_score(y_test, y_pred)}')
                print(f'AuC prob: {roc_auc_score(y_test, y_prob1)}')
                print(f'F1 score: {f1_score(y_test, y_pred)}')
                print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
                print(f'Precision: {precision_score(y_test, y_pred)}')
                print(f'Recall: {recall_score(y_test, y_pred)}')

                # show and save roc and precision-recall curves
                PrecisionRecallDisplay.from_predictions(y_test, y_pred)
                plt.show()
                PrecisionRecallDisplay.from_predictions(y_test, y_prob1)
                plt.savefig('C:/Users/yabertra/OneDrive - UGent/IoT PPM/test results/xgb 15 min 25 lags results PR.png')
                plt.show()

                RocCurveDisplay.from_predictions(y_test, y_pred)
                plt.show()
                RocCurveDisplay.from_predictions(y_test, y_prob1)
                plt.savefig('C:/Users/yabertra/OneDrive - UGent/IoT PPM/test results/xgb 15 min 25 lags results ROC.png')
                plt.show()

                ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                plt.savefig('C:/Users/yabertra/OneDrive - UGent/IoT PPM/test results/xgb 15 min 25 lags results CM.png')
                plt.show()

                # compute average time to adjustments

                test_log = pd.DataFrame()

                event_log = pd.read_csv('C:/Users/yabertra/OneDrive - UGent/IoT PPM/process_event_data2.csv',
                                        usecols=['CaseID', 'CompleteTimestamp', 'Vessel', 'ActivityID',
                                                                'lifecycle:transition'],
                                         parse_dates=['CompleteTimestamp'])
                event_log = event_log.loc[event_log['lifecycle:transition'] == 'complete']

                for batch in event_log['CaseID'].unique():
                    batch_event_log = event_log.loc[event_log['CaseID'] == batch]
                    batch_event_log.reset_index(inplace=True, drop=True)

                    batch_index = pd.MultiIndex.from_arrays(arrays=[np.full(len(batch_event_log),batch),batch_event_log.index], names=['CaseID', 'New Timestamp'])
                    batch_event_log.set_index(batch_index, inplace=True, drop=True)
                    batch_event_log = batch_event_log.copy()
                    batch_event_log.loc[:,'TimestampDif'] = batch_event_log['CompleteTimestamp'].diff()
                    batch_event_log.loc[:,'NextActivity'] = batch_event_log['ActivityID'].shift(-1)

                    batch_event_log.dropna(inplace=True)
                    test_log = pd.concat([test_log, batch_event_log], axis=0)

                test_log = test_log.loc[test_log.index.intersection(X_test.index)]
                test_log.sort_index(inplace=True)
                test_log.reset_index(inplace=True, drop=True)
                test_log['Prediction'] = y_pred#_sort
                test_log = test_log.loc[test_log['NextActivity'] == 'Pump adjustment']

                correct_prediction_log = test_log.loc[test_log['Prediction'] == 1]
                incorrect_prediction_log = test_log.loc[test_log['Prediction'] == 0]

                print(correct_prediction_log['TimestampDif'].mean())
                print(incorrect_prediction_log['TimestampDif'].mean())
                print(test_log['Prediction'].sum()/len(test_log))

                time_to_adjustments = pd.DataFrame(columns=['average time to correct adjustment', 'average time to wrong adjustment'])
                time_to_adjustments.loc[0,'average time to correct adjustment'] = correct_prediction_log['TimestampDif'].mean()
                time_to_adjustments.loc[0,'average time to wrong adjustment'] = incorrect_prediction_log['TimestampDif'].mean()
                time_to_adjustments.to_csv('C:/Users/yabertra/OneDrive - UGent/IoT PPM/test results/time_to_adjustments_iot_baseline.csv')
                print(time_to_adjustments)

evaluate_auc(iot_granularities, lag_windows, target='pump adjustment is next activity')