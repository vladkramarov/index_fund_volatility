import torch
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, QuantileLoss, Baseline, RMSE
from pytorch_forecasting.metrics import SMAPE, MAPE, MASE, RMSE
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import training.config
import data_processing.dataset_and_loaders as dataset_and_loaders
import training.callbacks as callbacks
from torch.utils.data import DataLoader
import warnings
import training.evaluator as evaluator
torch.manual_seed(42)
warnings.filterwarnings("ignore")  
import loader
import core


def get_trainer():
    early_stop_callback, lr_logger, logger, checkpoint = callbacks.get_callbacks()
    trainer = pl.Trainer(
        max_epochs=250,
        accelerator="cpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val = training.config.GRADIENT_CLIP_VAL,
        callbacks=[lr_logger, early_stop_callback, checkpoint],
        logger=logger, enable_checkpointing=True)

    return trainer

def get_tft_model(training_dataset: TimeSeriesDataSet):

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate= training.config.LEARNING_RATE,
        hidden_size= training.config.HIDDEN_SIZE, 
        attention_head_size= training.config.ATTENTION_HEADS, 
        dropout= training.config.DROPOUT,
        hidden_continuous_size= training.config.HIDDEN_CONTINUOUS_SIZE,
        output_size = 3, 
        loss= QuantileLoss([0.1, 0.5, 0.9]), 
        reduce_on_plateau_patience=5,
        optimizer=torch.optim.AdamW,
        log_interval=2)
    return tft


def run_training():
    training_dataset, val_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    trainer = get_trainer()
    tft = get_tft_model(training_dataset)
    trainer.fit(tft, train_dataloader, val_dataloader)
    return trainer, tft

if __name__ == "__main__":
    trainer, tft = run_training()
    training_dataset, val_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    test_dataset, test_dataloader = dataset_and_loaders.get_test_dataset_and_dataloaders(training_dataset)
    test_dataframe = loader.get_test_data()
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    test_results = evaluator.evaluate(trainer, best_tft, test_dataloader)
    preds, aligned = evaluator.predict_and_plot(best_tft, test_dataframe, test_dataset, 1)



