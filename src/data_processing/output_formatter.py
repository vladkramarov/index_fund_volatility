from __future__ import annotations
import pandas as pd
from pytorch_forecasting.models import base_model
import core
from typing import List
import src.training.config as config
import src.loader as loader


def generate_feature_names() -> List[str]:
    return [
        f"horizon_{horizon}_days"
        for horizon in range(1, config.MAX_PREDICTION_LENGTH + 1)
    ]

class OutputFormatter:
    def __init__(self, earliest_prediction_date: str, prediction, quantiles: List[float] = config.QUANTILES, prediction_data_type: str = "list"):
        self.prediction = prediction
        self.quantiles = quantiles
        self.quantile_results = {}
        self.generic_feature_names = generate_feature_names()
        self.earliest_prediction_date = earliest_prediction_date

        if isinstance(prediction, list):
            self.output = prediction[0].numpy()
            self.unaligned_results = prediction[-1]
        else:
            self.output = self.prediction[0].to("cpu").detach().numpy()
            self.unaligned_results = self.prediction.index.copy()
        # if prediction_data_type == "torch":
        #     self.unaligned_results = self.prediction.index.copy()
        #     self.output = self.prediction[0].to("cpu").detach().numpy()
        # else:
        #     self.unaligned_results = self.prediction[-1]
        #     self.output = self.prediction[0].numpy()

    def _process_results(self) -> pd.DataFrame:
        if self.quantiles:
            for position, quantile in enumerate(self.quantiles):
                quantile_predictions = self.output[:, :, position]
                quantile_feature_names = [
                    f"{feature_name}_P{quantile*100:.0f}"
                    for feature_name in self.generic_feature_names
                ]
                quantile_preds_df = pd.DataFrame(
                    quantile_predictions, columns=quantile_feature_names
                )
                self.unaligned_results[
                    quantile_feature_names
                ] = quantile_preds_df.values
                self.quantile_results[quantile] = quantile_preds_df
        else:
            self.unaligned_results[core.FORECAST_HORIZONS] = self.output
            self.unaligned_results = self.unaligned_results.sort_values(
                ["ticker", "idx"]
            )
        return self.unaligned_results

    def get_unaligned_results(self, original_data: pd.DataFrame) -> pd.DataFrame:
        self._process_results()
        self.unaligned_results = pd.merge(
            self.unaligned_results,
            original_data[["ticker", "idx", "date"]],
            on=["ticker", "idx"],
            how="left",
        )
        self.unaligned_results = self.unaligned_results.loc[
            self.unaligned_results["date"] >= self.earliest_prediction_date
        ]
        return self.unaligned_results

    def _shift_results(self, data) -> pd.DataFrame:
        for i in range(config.MAX_PREDICTION_LENGTH):
            data.iloc[:, i + 2] = data.iloc[:, i + 2].shift(i + 1)
        return data

    def get_last_result_per_ticker(self) -> pd.DataFrame:
        return self.unaligned_results.groupby("ticker").tail(1)

    def get_aligned_results(self, original_data: pd.DataFrame) -> pd.DataFrame:
        self._process_results()
        self.aligned_dataframes = {}
        for quantile in self.quantiles:
            columns = ["idx", "ticker"] + [
                col
                for col in self.unaligned_results.columns
                if f"P{quantile*100:.0f}" in col
            ]
            quantile_df = self.unaligned_results[columns]
            quantile_df = quantile_df.groupby("ticker").apply(self._shift_results)
            self.aligned_dataframes[quantile] = quantile_df

        self.aligned_results = pd.merge(
            self.aligned_dataframes[self.quantiles[0]],
            self.aligned_dataframes[self.quantiles[1]],
            on=["ticker", "idx"],
            how="left",
        )
        self.aligned_results = pd.merge(
            self.aligned_results,
            self.aligned_dataframes[self.quantiles[2]],
            on=["ticker", "idx"],
            how="left",
        )
        self.aligned_results = pd.merge(
            self.aligned_results,
            original_data[["ticker", "idx", "date"]],
            on=["ticker", "idx"],
            how="left",
        )
        return self.aligned_results
