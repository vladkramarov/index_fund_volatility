import torch
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, QuantileLoss, Baseline, RMSE
from pytorch_forecasting.metrics import SMAPE, MAPE, MASE, RMSE
import pandas as pd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import training.config
import data_processing.dataset_and_loaders as dataset_and_loaders
import training.callbacks as callbacks
from torch.utils.data import DataLoader
import warnings
import importlib
import core
importlib.reload(core)
importlib.reload(training.config)
import training.evaluator as evaluator
torch.manual_seed(42)
warnings.filterwarnings("ignore")  
import loader
import data_processing.output_formatter as output_formatter
import seaborn as sns
import matplotlib.dates as mdates
importlib.reload(output_formatter)
import joblib

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
        learning_rate= training.config.LEARNING_RATE*10,
        hidden_size= training.config.HIDDEN_SIZE, #13,
        attention_head_size= training.config.ATTENTION_HEADS, #6,
        dropout= training.config.DROPOUT,
        hidden_continuous_size= training.config.HIDDEN_CONTINUOUS_SIZE,#11,
        output_size = 3, 
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=5,
        # log_gradient_flow=True,
        log_interval=2)
    return tft


def run_training():
    training_dataset, val_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    trainer = get_trainer()
    tft = get_tft_model(training_dataset)
    trainer.fit(tft, train_dataloader, val_dataloader)
    return trainer, tft

def evaluate(trainer, tft, dataloader):
    return trainer.validate(tft, dataloader)

if __name__ == "__main__":
    trainer, tft = run_training()
    training_dataset, val_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()
    test_dataset, test_dataloader = dataset_and_loaders.get_test_dataset_and_dataloaders(training_dataset)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    # valid_results = evaluate(trainer, best_tft, val_dataloader)
    test_results = evaluate(trainer, best_tft, test_dataloader)
    test_data = loader.get_test_data()


best_model_path = 'lightning_logs/lightning_logs/version_175/checkpoints/epoch=21-step=9526.ckpt'
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
ts_dataset_params = best_tft.dataset_parameters

preds = best_tft.predict(test_dataset, return_index=True, return_x=True, mode='quantiles')

res = preds.index.copy()


formatter = output_formatter.OutputFormatter(preds)
unaligned = formatter.unaligned_results
unaligned
q = formatter.format_output_pipeline(test_data)
non_miss = q.loc[q['fh_10_0.9'].notnull()]
non_miss['date'] = pd.to_datetime(non_miss['date'])

def residual_plot(days_ahead: str = 'two', data: pd.DataFrame = non_miss):
    prediction_feature = f'{days_ahead}_days_ahead'
    residual = non_miss[prediction_feature] - non_miss[training.config.TARGET]
    plt.scatter(non_miss[training.config.TARGET], residual)
    plt.ylabel('Residual')
    plt.title(f'Residual plot for {days_ahead} days ahead')
    plt.show()
residual_plot('two')

def plot_results_over_time(days_ahead: int = 1, data: pd.DataFrame = non_miss):
    
    prediction_feature = f'fh_{days_ahead}_0.5'
    lower_limit = f'fh_{days_ahead}_0.1'
    upper_limit = f'fh_{days_ahead}_0.9'
    fig, ax = plt.subplots(4, 2, figsize=(15, 8))
    #plot results for each ticker in a separate subplot
    for i, ticker in enumerate(data['ticker'].unique()):
        ticker_data = data.loc[data['ticker'] == ticker]
        sns.lineplot(data=ticker_data, x='date', y=prediction_feature, ax=ax[i//2][i%2], legend='brief', label='prediction')
        #fill between the quantiles
        ax[i//2][i%2].fill_between(ticker_data['date'], ticker_data[lower_limit], ticker_data[upper_limit], alpha=0.3, label='0.1-0.9 quantile')
        sns.lineplot(data=ticker_data, x='date', y=training.config.TARGET, ax=ax[i//2][i%2])
        ax[i//2][i%2].set_title(f'{ticker} {days_ahead} days ahead')

        ax[i//2][i%2].xaxis.set_major_locator(mdates.DayLocator(interval=90))
    fig.tight_layout()
    plt.show()

plot_results_over_time(1)

test_df = loader.get_test_data()



# compare_density_plots('five')