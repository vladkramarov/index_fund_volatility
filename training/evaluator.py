
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pandas as pd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import training.config
import data_processing.dataset_and_loaders as dataset_and_loaders
from torch.utils.data import DataLoader
import core
from typing import List, Tuple, Dict
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, SCORERS
import matplotlib.dates as mdates
import data_processing.output_formatter as output_formatter
import loader

SCORERS['mae'] = mean_absolute_error
SCORERS['mape'] = mean_absolute_percentage_error
SCORERS['rmse'] = mean_squared_error

def get_ax(i, ax):
    return ax[i//2][i%2]

#REWRITE THIS GARBAGE
def get_metrics(data: pd.DataFrame, prediction_feature: str, target: str = training.config.TARGET, metric_list: List[str] = ['rmse', 'mae', 'mape']) -> Tuple[List[float], str]:
    rmse = mean_squared_error(data[target], data[prediction_feature], squared=False)
    mae = mean_absolute_error(data[target], data[prediction_feature])
    mape = mean_absolute_percentage_error(data[target], data[prediction_feature])
    text = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}'
    return [rmse, mae, mape], text

def plot_results_over_time(data: pd.DataFrame, days_ahead: int = 1):
    prediction_feature = f'fh_{days_ahead}_0.5'
    lower_limit = f'fh_{days_ahead}_0.1'
    upper_limit = f'fh_{days_ahead}_0.9'
    fig, ax = plt.subplots(4, 2, figsize=(15, 8))

    for i, ticker in enumerate(data['ticker'].unique()):
        ticker_data = data.loc[data['ticker'] == ticker]
        ax_current = get_ax(i, ax)
        sns.lineplot(data=ticker_data, x='date', y=prediction_feature, ax=ax_current, legend='brief', label='prediction')
        ax_current.fill_between(ticker_data['date'], ticker_data[lower_limit], ticker_data[upper_limit], alpha=0.3, label='0.1-0.9 quantile')
        sns.lineplot(data=ticker_data, x='date', y=training.config.TARGET, ax=ax_current)
        ax_current.set_title(f'{ticker} {days_ahead} days ahead')
        ax_current.xaxis.set_major_locator(mdates.DayLocator(interval=90))
        metrics, text = get_metrics(ticker_data, prediction_feature)
        ax_current.text(0.05, 0.95, text, transform=ax_current.transAxes, fontsize=9, verticalalignment='top')

    fig.tight_layout()
    plt.show()

def plot_metrics_over_horizons(data: pd.DataFrame, metric_name: str = 'mape'):
    metric = SCORERS[metric_name]
    horizon_scores = {}
    for horizon in range(1, training.config.MAX_PREDICTION_LENGTH + 1):
        prediction_feature = f'fh_{horizon}_0.5'
        score = metric(data[training.config.TARGET], data[prediction_feature])
        horizon_scores[prediction_feature] = score

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.barh(list(horizon_scores.keys()), list(horizon_scores.values()))

    plt.show()   

def evaluate(trainer: pl.Trainer, tft: TemporalFusionTransformer, dataloader: DataLoader) -> Dict[str, float]:
    return trainer.validate(tft, dataloader)

def predict_and_plot(model: TemporalFusionTransformer, original_dataframe: pd.DataFrame, dataset: TimeSeriesDataSet,days_ahead: int = 3) -> Tuple[List, pd.DataFrame]:
    preds = model.predict(dataset, return_index=True, return_x = True, mode='quantiles')
    output_form = output_formatter.OutputFormatter(earliest_prediction_date='2021-01-21', prediction=preds, prediction_data_type='list')
    aligned = output_form.get_aligned_results(original_dataframe)
    aligned = pd.merge(aligned, original_dataframe[['ticker', 'date', training.config.TARGET]], on=['ticker', 'date'])
    aligned.dropna(inplace=True)
    plot_results_over_time(aligned, days_ahead=days_ahead)
    plt.show()
    return preds, aligned

