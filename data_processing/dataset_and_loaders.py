import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, EncoderNormalizer
import core
import training.config
from typing import Tuple
import importlib
import numpy as np
import loader
from sklearn.preprocessing import StandardScaler, RobustScaler
importlib.reload(training.config)
import data_processing.data_formatter as data_formatter


def get_timeseries_datasets():
    train = loader.get_data_for_training()
    train = data_formatter.persist_column_type(train)
    train = train.loc[train['idx'] < 4500]
    training_dataset = TimeSeriesDataSet(
        train[lambda x: x.idx < 4000],
        time_idx="idx",
        target=training.config.TARGET,
        group_ids=training.config.GROUP_IDS,
        min_encoder_length=training.config.MAX_ENCODER_LENGTH - training.config.MAX_PREDICTION_LENGTH + 1,
        max_encoder_length=training.config.MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=training.config.MAX_PREDICTION_LENGTH,
        static_categoricals=training.config.STATIC_CATEGORICALS,
        time_varying_known_categoricals=training.config.TIME_VARYING_KNOWN_CATEGORICALS,
        # time_varying_unknown_categoricals=training.config.TIME_VARYING_UNKNOWN_CATEGORICALS,
        time_varying_known_reals=training.config.TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=training.config.TIME_VARYING_UNKNOWN_REALS,
        target_normalizer=GroupNormalizer(groups = training.config.GROUP_IDS, transformation='softplus'),
        add_relative_time_idx=True,
        add_encoder_length=True,
        add_target_scales=True,
        scalers={'robust': RobustScaler()},
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, train, min_prediction_idx = 4000, predict_mode=False, stop_randomization=True)
    return training_dataset, validation_dataset

def get_dataloaders(training_dataset, validation_dataset):
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=training.config.BATCH_SIZE, num_workers=4)
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=training.config.VAL_BATCH_SIZE, num_workers=10)
    
    return train_dataloader, val_dataloader

def get_timeseries_datasets_and_dataloaders():
    training_dataset, validation_dataset = get_timeseries_datasets()
    train_dataloader, val_dataloader = get_dataloaders(training_dataset, validation_dataset)
    return training_dataset, validation_dataset, train_dataloader, val_dataloader

def get_encoder_data(last_predicted_dataset:pd.DataFrame):
    encoder_start_idx = last_predicted_dataset.groupby('ticker')['idx'].max().min() - training.config.MAX_ENCODER_LENGTH
    return last_predicted_dataset.loc[last_predicted_dataset['idx'] > encoder_start_idx]

def get_extra_max_prediction_length_rows(data:pd.DataFrame):
    last_row_per_team = data.groupby('ticker').tail(1)
    return pd.concat(
        [last_row_per_team.assign(idx=lambda row: row.idx + i) for i in range(1, training.config.MAX_PREDICTION_LENGTH)])

def get_test_dataset_and_dataloaders(train_dataset: TimeSeriesDataSet):
    test_data = loader.get_test_data()
    valid_data = loader.get_valid_data()
    encoder_data = get_encoder_data(valid_data)
    extra_rows = get_extra_max_prediction_length_rows(test_data)
    test_data = pd.concat([encoder_data, test_data, extra_rows], axis=0, ignore_index=True)
    test_data = data_formatter.persist_column_type(test_data)
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_data, predict=False, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=training.config.VAL_BATCH_SIZE, num_workers = training.config.NUM_WORKERS, shuffle=False)
    return test_dataset, test_dataloader

