import datetime as dt
from typing import List, Dict, Union
import numpy as np
import importlib
import core
import loader
import training.config
import pandas as pd
import data_processing.data_formatter as data_formatter
import importlib
import datetime as dt
from pytorch_forecasting import TimeSeriesDataSet
import data_processing.input_validation as input_validation

importlib.reload(loader)



def get_last_market_date(date: Union[dt.datetime, str] = None, market_schedule: pd.DataFrame = None):
    '''Determines the last market date. Needed for validation purposes'''
    todays_date = pd.to_datetime(dt.datetime.today().strftime('%Y-%m-%d'))
    past_market_dates = market_schedule.loc[market_schedule['market_date'] < todays_date]
    last_market_date = pd.to_datetime(past_market_dates.tail(1).values[0][0])

    return todays_date, last_market_date, past_market_dates

def get_prediction_dates(last_market_date: Union[dt.datetime, str], market_schedule: pd.DataFrame) -> List[Union[dt.datetime, str]]:
    '''Needed for the decoder'''
    ...

class NewDataManager:

    def __init__(self, tickers: List[str], prediction_start_date: Union[dt.datetime, str]):
        self.tickers = tickers
        self.prediction_start_date = prediction_start_date
        self._market_schedule = loader.get_market_schedule()
        self._todays_date, self.last_market_date, self._past_market_dates = get_last_market_date(date = self.prediction_start_date, market_schedule=self._market_schedule)

    def _get_encoder_start_date(self):
        prediction_start_date = pd.to_datetime(self.prediction_start_date).strftime('%Y-%m-%d')
        business_days_to_subtract = training.config.MAX_ENCODER_LENGTH + training.config.VOLATILITY_WINDOW + 5
        self.encoder_start_date = pd.to_datetime(np.busday_offset(prediction_start_date, -business_days_to_subtract, roll='backward'))

    def _get_starting_idx_value(self):
        encoder_start_date_str = pd.to_datetime(self.encoder_start_date).strftime('%Y-%m-%d')
        business_days = np.busday_count(training.config.LAST_TEST_DATASET_DATE, encoder_start_date_str)
        self.starting_idx = business_days + training.config.LAST_TEST_IDX

    def _get_first_prediction_idx(self):
        if self.processed_data is None:
            raise ValueError('Must run process_new_data_pipeline() first')
        else:
            self.first_prediction_idx = self.processed_data.loc[self.processed_data['date']==self.prediction_start_date]['idx'].min()
    
    def _get_most_recent_idx(self):
        if self.processed_data is None:
            raise ValueError('Must run process_new_data_pipeline() first')
        else:
            self.most_recent_idx = self.processed_data['idx'].max()

    def _get_new_data(self):
        self.new_data = loader.get_data_from_api(start_date=self.encoder_start_date, tickers=self.tickers)
        self.new_data['Date'] = self.new_data['Date'].dt.tz_localize(None)
        self.new_data = self.new_data.loc[self.new_data['Date'] <= self.last_market_date]
        self.new_data['mask'] = 0
    
    def _get_future_dates_for_predictions(self):
        last_market_index = self._market_schedule.loc[self._market_schedule['market_date'] == self.last_market_date].index[0]
        self.future_prediction_dates = self._market_schedule.loc[last_market_index+1:last_market_index+training.config.MAX_PREDICTION_LENGTH-1]['market_date'].values

    def _get_future_rows(self):
        last_data_point_per_ticker = self.new_data.groupby('ticker').tail(1)
        self.future_rows = pd.concat([
            last_data_point_per_ticker.assign(Date = lambda x: self.future_prediction_dates[i]) for i in range(len(self.future_prediction_dates))
        ])
        self.future_rows['mask'] = 1
    
    def _add_future_rows(self):
        self._get_future_dates_for_predictions()
        self._get_future_rows()
        self.new_data = pd.concat([self.new_data, self.future_rows], axis=0)
    
    def _process_data(self):
        self.processed_data = data_formatter.data_management_pipeline(self.new_data, self.starting_idx)
        self.processed_data.rename(columns={'cpi_date': 'cpi_report'}, inplace=True)
        self.processed_data = data_formatter.persist_column_type(self.processed_data)
        
    def process_new_data_pipeline(self):
        self._get_encoder_start_date()
        self._get_starting_idx_value()
        self._get_new_data()
        self._add_future_rows()
        self._process_data()
        self.processed_data['volatility_target'] = 0.1 
        return self.processed_data


def new_data_pipeline(tickers: List[str], prediction_start_date: Union[dt.datetime, str]):
    data_manager = NewDataManager(tickers, prediction_start_date)
    processed_dataframe = data_manager.process_new_data_pipeline()
    return processed_dataframe, data_manager

def get_timeseries_dataset(processed_data: pd.DataFrame):
    ts_dataset_params = loader.get_timeseries_params()
    processed_ts_dataset = TimeSeriesDataSet.from_parameters(ts_dataset_params, processed_data, predict = False)
    return processed_ts_dataset





new = NewDataManager(['AAPL'], '2023-08-05')
processed = new.process_new_data_pipeline()

