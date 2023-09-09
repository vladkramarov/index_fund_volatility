import pandas as pd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import seaborn as sns
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from torch.utils.data import DataLoader
import matplotlib.dates as mdates
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    SCORERS,
)
import src.training.config as config
import src.data_processing.dataset_and_loaders as dataset_and_loaders
import core
import data_processing.output_formatter as output_formatter
import src.loader as loader

SCORERS["mae"] = mean_absolute_error
SCORERS["mape"] = mean_absolute_percentage_error
SCORERS["rmse"] = mean_squared_error


def get_ax(i, ax):
    return ax[i // 2][i % 2]


def evaluate(
    trainer: pl.Trainer, tft: TemporalFusionTransformer, dataloader: DataLoader
) -> Dict[str, float]:
    return trainer.validate(tft, dataloader)


def get_metrics(
    data: pd.DataFrame,
    prediction_feature: str,
    target: str = config.TARGET,
    metric_list: List[str] = ["rmse", "mae", "mape"],
) -> Tuple[List[float], str]:
    rmse = mean_squared_error(data[target], data[prediction_feature], squared=False)
    mae = mean_absolute_error(data[target], data[prediction_feature])
    mape = mean_absolute_percentage_error(data[target], data[prediction_feature])
    text = f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}"
    return [rmse, mae, mape], text


def plot_results_over_time(
   aligned_results: pd.DataFrame, days_ahead: int = 1, overwrite_in_the_folder: bool = False
):
    prediction_feature = f"horizon_{days_ahead}_days_P50"
    lower_limit = f"horizon_{days_ahead}_days_P10"
    upper_limit = f"horizon_{days_ahead}_days_P90"
    fig, ax = plt.subplots(4, 2, figsize=(15, 8))

    for i, ticker in enumerate(aligned_results["ticker"].unique()):
        ticker_data =aligned_results.loc[aligned_results["ticker"] == ticker]
        ax_current = get_ax(i, ax)
        sns.lineplot(
           aligned_results=ticker_data,
            x="date",
            y=prediction_feature,
            ax=ax_current,
            legend="brief",
            label="prediction",
        )
        ax_current.fill_between(
            ticker_data["date"],
            ticker_data[lower_limit],
            ticker_data[upper_limit],
            alpha=0.3,
            label="0.1-0.9 quantile",
        )
        sns.lineplot(
           aligned_results=ticker_data, x="date", y=config.TARGET, ax=ax_current
        )
        ax_current.set_title(f"{ticker} {days_ahead} days ahead")
        ax_current.xaxis.set_major_locator(mdates.DayLocator(interval=90))
        metrics, text = get_metrics(ticker_data, prediction_feature)
        ax_current.text(
            0.05,
            0.95,
            text,
            transform=ax_current.transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    fig.tight_layout()

    if overwrite_in_the_folder:
        fig.savefig(f"{core.IMAGE_PATH}/results_over_time_{days_ahead}_days_ahead.png")

    plt.show()
    return fig, ax


def plot_metrics_over_horizons(
    aligned_results: pd.DataFrame, metric_name: str = "mape", overwrite_in_the_folder: bool = False
):
    metric = SCORERS[metric_name]
    horizon_scores = {}
    for horizon in range(1, config.MAX_PREDICTION_LENGTH + 1):
        prediction_feature = f"horizon_{horizon}_days_P50"
        score = metric(aligned_results[config.TARGET], aligned_results[prediction_feature])
        horizon_scores[prediction_feature] = score

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_title(f"{(str.upper(metric_name))} over Prediction Horizons")
    ax.barh(list(horizon_scores.keys()), list(horizon_scores.values()))

    if overwrite_in_the_folder:
        fig.savefig(f"{core.IMAGE_PATH}/metrics_over_horizons_{metric_name}.png")

    plt.show()

    return fig, ax


def predict_and_plot(
    model: TemporalFusionTransformer,
    original_dataframe: pd.DataFrame,
    dataset: TimeSeriesDataSet,
    days_ahead: int = 3,
) -> Tuple[List, pd.DataFrame]:
    preds = model.predict(dataset, return_index=True, return_x=True, mode="quantiles")
    output_form = output_formatter.OutputFormatter(
        earliest_prediction_date="2021-01-21",
        prediction=preds,
        prediction_data_type="list",
    )
    aligned = output_form.get_aligned_results(original_dataframe)
    aligned = pd.merge(
        aligned,
        original_dataframe[["ticker", "date", config.TARGET]],
        on=["ticker", "date"],
    )
    aligned.dropna(inplace=True)
    fig, ax = plot_results_over_time(aligned, days_ahead=days_ahead)
    plt.show()
    return preds, aligned


def plot_attention_pattern(model: TemporalFusionTransformer, raw_predictions: List):
    interpretation_data = model.interpret_output(raw_predictions[0], reduction="sum")
    intepretation = model.plot_interpretation(interpretation_data)
    plt.show()
    return intepretation
