import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import warnings
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
)
import src.loader as loader
import src.training.config as config
import src.data_processing.dataset_and_loaders as dataset_and_loaders
import src.training.callbacks as callbacks
import src.training.evaluator as evaluator
torch.manual_seed(42)
warnings.filterwarnings("ignore")


def get_trainer():
    early_stop_callback, lr_logger, logger, checkpoint, model_summary = callbacks.get_callbacks()
    trainer = pl.Trainer(
        max_epochs=250,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        callbacks=[lr_logger, early_stop_callback, checkpoint, model_summary],
        logger=logger,
        enable_checkpointing=True,
    )

    return trainer


def get_tft_model(training_dataset: TimeSeriesDataSet):
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.LEARNING_RATE,
        hidden_size=config.HIDDEN_SIZE,
        attention_head_size=config.ATTENTION_HEADS,
        dropout=config.DROPOUT,
        hidden_continuous_size=config.HIDDEN_CONTINUOUS_SIZE,
        output_size=3,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=5,
        optimizer=torch.optim.AdamW,
        log_interval=2,
    )
    return tft


def run_training():
    training_dataset, val_dataset, train_dataloader, val_dataloader \
        = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    trainer = get_trainer()
    tft = get_tft_model(training_dataset)
    trainer.fit(tft, train_dataloader, val_dataloader)
    return trainer, tft


if __name__ == "__main__":
    trainer, tft = run_training()
    training_dataset, val_dataset, train_dataloader, val_dataloader = \
        dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    test_dataset, test_dataloader = \
        dataset_and_loaders.get_test_dataset_and_dataloaders(training_dataset)
    test_dataframe = loader.get_test_data()
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    test_results = evaluator.evaluate(trainer, best_tft, test_dataloader)
    preds, aligned = evaluator.predict_and_plot(
        best_tft, test_dataframe, test_dataset, 3
    )
