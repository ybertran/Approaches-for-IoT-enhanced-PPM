from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
from os import path
import seaborn as sns
import matplotlib.pyplot as plt

iot_granularities = ['1h', '15min', '5min', '1min'] # granularity level at which IoT data are aggregated
lag_windows = [5, 10, 25, 50, 100] # number of lags to add to the input data (= number of time steps model can see in the past)
time_horizon = ['15min', '30min', '1h', '2h'] # time window in which next pump adjustment should occur
targets = ['pump adjustment present in next time window', 'pump adjustment is next activity']
directory_paths = ['xgb results/', 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/xgb results/']

for target in targets:
    print('############################################################################################################')
    for lag_window in lag_windows:
        print('-------------------------------------------------------------------------------------------------------------')

        data = {
            "iot_granularity": [],
            "horizon": [],
            "f1_score": []
        }

        for iot_granularity in iot_granularities:

            for horizon in time_horizon:
                if pd.Timedelta(horizon) >= pd.Timedelta(iot_granularity):
                    if target == 'pump adjustment present in next time window':
                        file_path = 'xgb results/model ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows ' + horizon + ' time horizon/confusion_matrix_with_report.csv'
                    else:
                        file_path = 'C:/Users/yabertra/OneDrive - UGent/IoT PPM/xgb results/model ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows 24h time horizon ' + target + ' target/confusion_matrix_with_report.csv'
                    #file_path = directory_path + 'model ' +  iot_granularity + ' iot granularity - ' + str(lag_window) + ' lag windows ' + horizon + ' time horizon/confusion_matrix_with_report.csv'
                    if path.exists(file_path):

                        cm_df = pd.read_csv(file_path, index_col='Unnamed: 0')
                        print(cm_df)
                        tp = cm_df.loc['True_1', 'Pred_1']
                        fp = cm_df.loc['True_0', 'Pred_1']
                        fn = cm_df.loc['True_1', 'Pred_0']
                        f1_score = (2*tp)/(2*tp+fp+fn)
                        data['iot_granularity'].append(iot_granularity)
                        data['horizon'].append(horizon)
                        data['f1_score'].append(f1_score)
                        print(f'F1 score for model with {iot_granularity} iot granularity, {str(lag_window)} lag windows, {horizon} time horizon and {target} target: {f1_score}')

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