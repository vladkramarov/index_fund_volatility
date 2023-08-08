from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import core
import importlib
import loader
import pandas as pd
import joblib
import data_processing.new_data_processor as new_data_processor
import data_processing.output_formatter as output_formatter
importlib.reload(core)

def predict(processed_data: pd.DataFrame):
    processed_data['date'] = processed_data['date'].astype(str)
    model = loader.get_model()
    ts_dataset_params = loader.get_timeseries_params()
    ts_dataset = TimeSeriesDataSet.from_parameters(ts_dataset_params, processed_data, predict=False)
    preds = model.predict(ts_dataset, return_index=True, return_x=True, mode='quantiles')
    return preds

def process_output(preds, prediction_start_date: str, processed_data: pd.DataFrame) -> pd.DataFrame:
    output_form = output_formatter.OutputFormatter(earliest_prediction_date=prediction_start_date, prediction=preds)
    return output_form.get_unaligned_results(processed_data)
    
def prediction_pipeline(tickers = ['AAPL', 'MSFT'], prediction_start_date = '2023-07-25') -> pd.DataFrame:
    processed_data, _ = new_data_processor.new_data_pipeline(tickers = tickers, prediction_start_date = prediction_start_date)
    processed_data['volatility_target'] = 0.01
    preds = predict(processed_data)
    return process_output(preds, prediction_start_date, processed_data)

