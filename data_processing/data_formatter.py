import yfinance as yf
import datetime as dt
from typing import List, Dict, Union
import numpy as np
import importlib
from typing import Callable
import core
import loader
import training.config
import pandas as pd
importlib.reload(training.config)
import matplotlib.pyplot as plt
COLUMNS_TO_DROP = ['open', 'dividends', 'stock splits']

class DataProcessor:
    def __init__(self,  data: pd.DataFrame,
                 identifier_column: str = 'ticker', volatility_window: int = training.config.VOLATILITY_WINDOW):
        self.data = data.reset_index().rename(columns={'index': 'idx'})
        self.identifier = identifier_column
        self.volatility_window = volatility_window
        
    def format_column_names(self):
        self.data.columns = self.data.columns.str.lower()

    def create_date_columns(self):
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_the_week'] = self.data['date'].dt.dayofweek

    def fix_idx_column(self, starting_index: int = 0):
        self.data = self.data.sort_values(['ticker', 'date'])
        self.data['idx'] = self.data.groupby('ticker')['idx'].cumcount() + starting_index

    def create_log_returns(self, close_price: str = 'close'):
        '''Used to fill blank values for new data'''
        self.data['log_close'] = np.log(self.data[close_price])
        self.data['log_returns'] = self.data.groupby(self.identifier)['log_close'].diff()
        self.data['log_returns_absolute'] = self.data['log_returns'].abs()
        
    def create_volatility(self):
        if 'log_returns' in self.data.columns:
            self.data['historical_volatility'] = self.data.groupby(self.identifier)['log_returns'].rolling(self.volatility_window).std().reset_index(0, drop=True)
        else:
            raise ValueError('Must create log returns before calculating volatility!')

    def create_previous_day_range(self, high_price: str = 'high', low_price: str = 'low'):
        if high_price and low_price in self.data.columns:
            self.data['prev_day_range'] = self.data.groupby(self.identifier)[high_price].shift(1) - self.data.groupby(self.identifier)[low_price].shift(1)

    def fill_blank_numerical_cols(self):
        # self.data['log_returns_squared'] = self.data['log_returns_squared'].fillna(self.get_log_returns_squared())
        self.data['prev_day_range'] = self.data['prev_day_range'].fillna(self.get_previous_day_range())
        # self.data['moving_avg50_200'] = self.data['moving_avg50_200'].fillna(self.get_moving_avg_ratio())

    def drop_unused_columns(self, columns_to_drop: List[str]):
        self.data.drop(columns=columns_to_drop, inplace=True)

    def drop_blank_rows(self):
        self.data.dropna(axis=0, inplace=True)
    
    def add_earnings_dates(self):
        earnings_data = pd.read_csv(core.EARNINGS_DATA)
        earnings_data['earnings_date'] = pd.to_datetime(earnings_data['earnings_date'])
        self.data = pd.merge(self.data, earnings_data[['earnings_date', 'ticker']], left_on=['date', 'ticker'], right_on=['earnings_date', 'ticker'], how = 'left')
        self.data['earnings_date'] = self.data['earnings_date'].apply(lambda x: 0 if pd.isnull(x) else 1)
    def add_cpi_report(self):
        cpi_data = pd.read_csv(core.CPI_DATA)
        cpi_data['cpi_date'] = pd.to_datetime(cpi_data['cpi_date'])
        self.data = pd.merge(self.data, cpi_data[['cpi_date']], left_on='date', right_on='cpi_date', how = 'left')
        self.data['cpi_date'] = self.data['cpi_date'].apply(lambda x: 0 if pd.isnull(x) else 1)

    def positive_or_negative(self):
        self.data['positive_or_negative'] = self.data['log_returns'].apply(lambda row: 1 if row > 0 else 0)


def data_management_pipeline(data, starting_idx_value: int = 0):
    dm = DataProcessor(data)
    dm.format_column_names()
    dm.create_date_columns()
    dm.create_log_returns()
    dm.create_volatility()
    dm.create_previous_day_range()
    dm.drop_unused_columns(COLUMNS_TO_DROP)
    # dm.fill_blank_numerical_cols()
    dm.add_earnings_dates()
    dm.add_cpi_report()
    dm.positive_or_negative()
    dm.drop_blank_rows()
    dm.fix_idx_column(starting_idx_value)

    return dm.data

def split_train_valid_test(data:pd.DataFrame):
    max_idx = data['idx'].max()
    train_end_idx = max_idx - training.config.VALIDATION_DATA_LENGTH - training.config.TEST_DATA_LENGTH
    valid_end_idx = max_idx - training.config.TEST_DATA_LENGTH
    train = data[data['idx'] <= train_end_idx]
    valid = data[(data['idx'] > train_end_idx) & (data['idx'] <= valid_end_idx)]
    test = data[data['idx'] > valid_end_idx]
    return train, valid, test

def save_train_valid_test(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame):
    train_data.to_csv(core.TRAIN_DATA_PATH, index = False)
    valid_data.to_csv(core.VALID_DATA_PATH, index = False)
    test_data.to_csv(core.TEST_DATA_PATH, index = False)

def persist_column_type(data):
    data[training.config.TIME_VARYING_KNOWN_CATEGORICALS+training.config.TIME_VARYING_UNKNOWN_CATEGORICALS]\
          = data[training.config.TIME_VARYING_KNOWN_CATEGORICALS+training.config.TIME_VARYING_UNKNOWN_CATEGORICALS].astype(str)
    return data

def get_future_dates_for_prediction(data):
    last_date = data.groupby('ticker')['date'].max().reset_index()