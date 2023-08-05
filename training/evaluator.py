import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, QuantileLoss, Baseline
from pytorch_forecasting.metrics import SMAPE, MAPE, MASE, RMSE
import pandas as pd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import training.config
import data_processing.dataset_and_loaders as dataset_and_loaders
import training.callbacks as callbacks
from torch.utils.data import DataLoader
import core
from typing import List

class Evaluator:

    def __init__(self, output_data: pd.DataFrame, tickers: List[str] = core.TICKERS, target_feature: str = training.config.TARGET) -> None:
        self.output_data = output_data
        self.target_feature = target_feature
        self.tickers = tickers
    

    
    
    def plot_predictions_per_ticker(self, start_idx:int = 0):

        if self.val_dataset is None:
            self.train_dataset, self.val_dataset = dataset_and_loaders.get_timeseries_datasets()
        
        for ticker in self.tickers:
            predictions = self.tft_model.predict(
                self.val_dataset.filter(lambda x: (x.ticker == ticker)), return_x=True, mode='raw')
            self.tft_model.plot_prediction(predictions.x, predictions.output, idx = start_idx)
        plt.show()

    def get_ts_datasets(self):
        self.train_dataset, self.val_dataset = dataset_and_loaders.get_timeseries_datasets()

    def get_loss_metric_per_ticker(self):
        ...

# def get_raw_predictions(trainer: pl.Trainer, val_dataloader: torch.utils.data.DataLoader):
#     best_model_path = trainer.checkpoint_callback.best_model_path
#     best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
#     raw_predictions = best_tft.predict(val_dataloader, mode='raw', return_x=True)
#     return raw_predictions

# best_model_path = trainer.checkpoint_callback.best_model_path
# best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)



# raw_predictions = best_tft.predict(val_dataloader, mode='raw', return_x=True)
# raw_predictions.output[0].shape

# #use raw_predictions to predict plot predictions per group
# for idx in range(10):
#     best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)

# plt.show()


# b = trainer.validate(tft, val_dataloader, ckpt_path=trainer.checkpoint_callback.best_model_path)

