import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import datetime

#import dask.dataframe as dd
#import dask.delayed as delayed
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
import torch


class DataLoader:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def resolve_duplicates(group):
        """
        Resolve duplicate timestamps by taking the ceil of the timestamp
        """
        if group["concept:name"].nunique() > 1:
            warnings.warn(
                f"Found multiple events in the same minute: \n {group[['concept:name', 'og_timestamp']]}"
            )
            for j in range(len(group)):
                # Convert the timestamp to a numerical representation
                new_timestamp = group.iloc[
                    j, group.columns.get_loc("og_timestamp")
                ] + datetime.timedelta(seconds=j)
                group.iloc[j, group.columns.get_loc("timestamp")] = pd.to_datetime(
                    new_timestamp
                ).round(f"1S")
                print(
                    f"Instead took the ceil of the timestamp: {group.iloc[j, group.columns.get_loc('timestamp')]}"
                )
            return group
        else:
            return group

    def load_sensors(self, sensor_path, subset=None, per_minute=False):
        """
        Load sensor data using pandas
        :param sensor_path: path to the sensor data
        :param subset: subset of batches to load (if you wanna experiment with fewer batches, see example in notebook)
        :param per_minute: whether to aggregate the data per minute
        """
        batch_files = glob.glob(os.path.join(self.path, sensor_path))
        batch_files_with_ids = [
            (file, os.path.basename(file).split()[2].split(".")[0])
            for file in batch_files
        ]
        if subset is not None:
            subset = set(subset)
            batch_files_with_ids = [
                (file, id_) for file, id_ in batch_files_with_ids if id_ in subset
            ]
        dfs = []
        for file, batch_id in batch_files_with_ids:
            dtype = {
                "New Timestamp": str,
                "Filter 1 DeltaP": float,
                "Filter 1 Inlet Pressure": float,
                "Filter 2 DeltaP": float,
                "Pump Circulation Flow": float,
                "Tank Pressure": float,
            }
            df = pd.read_csv(file, sep=";", dtype=dtype)
            df["batch_id"] = batch_id
            dfs.append(df)
        self.sensor_data = pd.concat(dfs, ignore_index=True)
        self.sensor_data["timestamp"] = pd.to_datetime(
            self.sensor_data["New Timestamp"]
        )
        self.sensor_data.drop(columns=["New Timestamp"], inplace=True)
        if per_minute:
            self.sensor_data = (
                self.sensor_data.groupby(
                    by=["batch_id", pd.Grouper(key="timestamp", freq="1Min")]
                )
                .mean()
                .reset_index()
            )
        self.sensor_data["time_idx"] = (
            self.sensor_data.groupby(["batch_id"]).cumcount().values
        )
        self.sensor_data.set_index(["batch_id", "timestamp"], inplace=True)
        return self.sensor_data

    def load_events(self, event_path):
        """
        Load event data
        """
        self.event_data = pd.read_csv(
            os.path.join(self.path, event_path), sep=";", index_col=0
        )
        self.event_data = self.event_data.drop_duplicates()
        self.event_data.rename(
            columns={
                "case:concept:name": "batch_id",
                "time:timestamp": "timestamp",
            },
            inplace=True,
        )
        self.event_data["timestamp"] = pd.to_datetime(self.event_data["timestamp"])
        self.event_data.set_index(["batch_id", "timestamp"], inplace=True)
        return self.event_data

    @staticmethod
    def add_lags(df, group_col, lag_columns, num_lags, print_msg=False):
        """
        Add lags to the data
        :param df: dataframe to add lags to
        :param group_col: column to group by
        :param lag_columns: columns to add lags to
        :param num_lags: number of lags to add
        :param print_msg: whether a message should be printed when lags are added to the df
        """
        lag_dict = {}

        for col in lag_columns:
            for lag in range(1, num_lags + 1):
                if print_msg:
                    print(f"Adding lag {lag} for column {col}")
                if group_col == 'index':
                    lag_dict[f"{col}_lag_{lag}"] = df[col].shift(
                        lag
                    )
                else:
                    df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(
                        lag
                    )  # , meta=(f"{col}_lag_{lag}", "float64"))
            #df = df.drop(columns=[col])
        lag_df = pd.DataFrame(lag_dict)
        df = pd.concat([df,lag_df], axis=1)
        return df

    @staticmethod
    def merge_event_min_to_list(event_data):
        """
        Merge events that happen in the same minute to a list
        """
        event_data = event_data.copy()
        event_data["timestamp"] = (
            pd.to_datetime(event_data["timestamp"]).dt.round("1min").values
        )
        event_data = (
            event_data.groupby(["batch_id", "timestamp"])["concept:name"]
            .unique()
            .apply(list)
            .reset_index()
        )
        return event_data

    @staticmethod
    def merge_event_seconds(event_data):
        """
        Resolve situations where there are multiple events within a second by applying the 'resolve duplicates' function
        """
        event_data = event_data.copy()
        event_data["og_timestamp"] = event_data["timestamp"]
        event_data["timestamp"] = (
            pd.to_datetime(event_data["timestamp"]).dt.round("1S").values
        )
        event_data = event_data.groupby(["batch_id", "timestamp"]).apply(
            DataLoader.resolve_duplicates
        )
        return event_data

    @staticmethod
    def merge_sensors_and_events(sensor_data, event_data):
        """
        Merge sensor and event data
        """
        merged_data = pd.merge(
            sensor_data,
            event_data,
            how="left",
            on=["batch_id", "timestamp"],
        )
        return merged_data

    @staticmethod
    def check_cycle(sequence, start="Pump start", stop="Pump stop"):
        """
        Check if we're in a 'valid' cycle (i.e. between pump start and pump stop) and tag those as 'in_cycle'
        Otherwise it's tagged as 'out_cycle' meaning that it's outside of a pump start/pump stop
        """
        n = len(sequence)
        in_cycle_status = False
        start_seen = False
        statuses = pd.Series(["out_cycle"] * n, index=sequence.index)

        for i in sequence.index:
            if sequence.loc[i] == start:
                in_cycle_status = True
                start_seen = True
            elif sequence.loc[i] == stop:
                if in_cycle_status and start_seen:
                    in_cycle_status = False
                    statuses.loc[i] = "in_cycle"
                start_seen = False
            if in_cycle_status and start_seen:
                statuses.loc[i] = "in_cycle"

        return statuses

    @staticmethod
    def check_time_diff(sequence, threshold="10 min"):
        """
        Helper function to check if the time difference between two events is less than a threshold
        """
        return sequence < pd.Timedelta(threshold)

    @staticmethod
    def filter_events(data, start_event, group_column, threshold="10 min"):
        """
        Check if an event is within a certain time window of a start event
        """
        df = data.copy()
        # Create a boolean mask for start events
        mask_start = df["concept:name"] == start_event
        df.loc[mask_start, "last_start"] = df.loc[mask_start, "timestamp"]
        df["last_start"] = df.groupby(group_column)["last_start"].ffill()
        # Calculate the time difference between adjustment events and the last seen start event
        df["time_diff"] = df["timestamp"] - df["last_start"]
        df["status"] = df.groupby(group_column)["concept:name"].apply(
            lambda x: DataLoader.check_cycle(x, "Pump start", "Pump stop")
        )
        pump_adj = df["concept:name"] == "Pump adjustment"
        in_start_window = DataLoader.check_time_diff(
            df["time_diff"], threshold=threshold
        )
        df.loc[(pump_adj & in_start_window), "concept:name"] = "Start adjustment"
        return df

    @staticmethod
    def plot_sample_timediff(
        data, time_diff, concept_name, group_column, sample_size=10
    ):
        mask = data["concept:name"] == concept_name
        data = data[mask]
        random_sample = np.random.choice(
            data[group_column].unique(), size=sample_size, replace=False
        )
        sub_group = data[data[group_column].isin(random_sample)]
        sub_group["time_diff"] = sub_group["time_diff"].dt.total_seconds() / 60
        sub_group.boxplot(column=time_diff, by=group_column, rot=90)
        plt.xlabel(group_column)
        plt.ylabel("Time diff since Start pump (min)")
        plt.title("Boxplot - Time diff since Start pump")
        plt.show()


class TimeSeriesDataLoader:
    """
    Class to load data into a TimeSeriesDataSet from pytorch-lightning
    """

    def __init__(
        self,
        data,
        time_idx,
        target_cols,
        group_cols=None,
        time_varying_unknown_reals=None,
        **kwargs,
    ):
        self.train_data = TimeSeriesDataSet(
            data=data,
            time_idx=time_idx,
            target=target_cols,
            group_ids=group_cols,
            time_varying_unknown_reals=time_varying_unknown_reals,
            **kwargs,
        )
        self.batch_size = kwargs.get("batch_size", 64)
        self.shuffle = kwargs.get("shuffle", True)

    def get_validation_data(self, val_df, batch_size):
        val = TimeSeriesDataSet.from_dataset(
            self.train_data, val_df, predict=False, stop_randomization=True
        )
        val_dataloader = val.to_dataloader(
            train=False,
            batch_size=batch_size * 10,
        )
        return val_dataloader

    def get_test_data(self, test_df, batch_size):
        test = TimeSeriesDataSet.from_dataset(
            self.train_data, test_df, predict=True, stop_randomization=True
        )
        test_dataloader = test.to_dataloader(
            train=False,
            batch_size=batch_size * 10,
        )
        return test_dataloader

    def get_data_loaders(self, train_val_test=[0.7, 0.15, 0.15], **kwargs):
        lengths = [int(len(self.dataset) * i) for i in train_val_test]
        lengths.append(len(self.dataset) - sum(lengths))
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, lengths
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, **kwargs
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, **kwargs
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, **kwargs
        )
        return train_loader, val_loader, test_loader
