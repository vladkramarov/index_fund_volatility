from __future__ import annotations
import pandas as pd
import core
import importlib
from typing import List
import training.config
from pytorch_forecasting.models import base_model
import loader
importlib.reload(core)

def generate_feature_names() -> List[str]:
    return [f'fh_{horizon}' for horizon in range(1, training.config.MAX_PREDICTION_LENGTH + 1)]

class OutputFormatter:
    def __init__(self, earliest_prediction_date: str, prediction, quantiles: List[float] = training.config.QUANTILES):
        self.prediction = prediction
        self.quantiles = quantiles
        self.quantile_results = {}
        self.generic_feature_names = generate_feature_names()
        self.earliest_prediction_date = earliest_prediction_date

        if isinstance(self.prediction, list):
            self.unaligned_results = self.prediction[-1]
            self.output = self.prediction[0].to('cpu').detach().numpy()
        else:
            self.unaligned_results = self.prediction[0].to('cpu').numpy()
            self.output = self.prediction.index.copy()

        
    def _process_results(self) -> pd.DataFrame:
        if self.quantiles:
            for position, quantile in enumerate(self.quantiles):
                quantile_predictions = self.output[:, :, position]
                quantile_feature_names = [f'{feature_name}_{quantile}' for feature_name in self.generic_feature_names]
                quantile_preds_df = pd.DataFrame(quantile_predictions, columns=quantile_feature_names)
                self.unaligned_results[quantile_feature_names] = quantile_preds_df.values
                self.quantile_results[quantile] = quantile_preds_df
        else:
            self.unaligned_results[core.FORECAST_HORIZONS] = self.output
            self.unaligned_results = self.unaligned_results.sort_values(['ticker', 'idx'])
        return self.unaligned_results
    
    def get_unaligned_results(self, original_data: pd.DataFrame) -> pd.DataFrame:
        self.unaligned_results = self.prediction[-1]
        self._process_results()
        self.unaligned_results = pd.merge(self.unaligned_results, original_data[['ticker', 'idx', 'date']], on = ['ticker', 'idx'], how = 'left')
        self.unaligned_results = self.unaligned_results.loc[self.unaligned_results['date'] >= self.earliest_prediction_date]
        return self.unaligned_results

    def shift_results(self, data) -> pd.DataFrame:
        for i in range(training.config.MAX_PREDICTION_LENGTH):
            data.iloc[:, i+2] = data.iloc[:, i+2].shift(i)
        return data
    
    def get_last_result_per_ticker(self) -> pd.DataFrame:
        return self.unaligned_results.groupby('ticker').tail(1)

    def get_aligned_results(self) -> pd.DataFrame:
        self._process_results()
        self.aligned_results = self.unaligned_results.groupby('ticker').apply(self.shift_results).reset_index(drop=True)
        return self.aligned_results
    
    def merge_columns(self, original_data: pd.DataFrame, columns_to_merge: List[str] = core.ADDITIONAL_OUTPUT_COLS) -> pd.DataFrame:
        self.final_results = pd.merge(left = self.aligned_results, right = original_data[columns_to_merge], on = ['ticker', 'idx'], how = 'left')
        return self.final_results

    def format_output_pipeline(self, original_data) -> pd.DataFrame:
        self.get_aligned_results()
        self.merge_columns(original_data = original_data)
        return self.final_results
    