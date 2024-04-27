import glob
import os
import numpy as np
import pandas as pd

# custom modules
from data_loader import DataLoader


# set random seed
rng = np.random.default_rng(96)


def process_data(sample: bool = False, sample_size: int = 50):
    """
    Loads and processes the data from the raw data folder and saves it to the processed data folder.
    :param sample: Whether to sample the data or not.
    :param sample_size: The size of the sample.
    :return: None
    """
    all_batches = []
    data_dir = os.path.join("..", "data", "raw")
    sensor_data_dir = os.path.join(data_dir, "sensor_data_second_JSR", "*.csv")
    output_dir = os.path.join("..", "processed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "merged_data.csv")

    for i in glob.glob(sensor_data_dir):
        all_batches.append(os.path.basename(i).split()[2].split(".")[0])

    if sample:
        batch_subset = rng.choice(all_batches, sample_size, replace=False)
    else:
        batch_subset = None

    dl = DataLoader("../data/raw/")
    sensor_data = dl.load_sensors(
        "sensor_data_second_JSR/*.csv", subset=batch_subset, per_minute=True
    )
    event_data = dl.load_events("Event log.csv")

    events_clean = dl.filter_events(event_data.reset_index(), "Pump start", "batch_id")
    events_clean_in_cycle = events_clean[events_clean["status"] == "in_cycle"]
    events_clean_in_cycle = dl.merge_event_min_to_list(events_clean_in_cycle)
    events_clean_in_cycle.set_index(["batch_id", "timestamp"], inplace=True)

    events_clean_in_cycle["concept:name"] = events_clean_in_cycle[
        "concept:name"
    ].astype(str)
    merged_data = dl.merge_sensors_and_events(sensor_data, events_clean_in_cycle)
    get_events_target(merged_data)
    merged_data["concept:name"].fillna(
        "censored event", inplace=True
    )  # []get_events_target(merged_data)
    merged_data.to_csv(output_path)


def get_events_target(df):
    """
    Creates a target variable for the events.
    :param df: The dataframe containing the events.
    :return: None
    """
    df["target"] = (
        0
        - df["concept:name"].isna().astype(int)
        + df["concept:name"].fillna("other").str.contains("pump adjustment").astype(int)
    )


if __name__ == "__main__":
    process_data(sample=False)
