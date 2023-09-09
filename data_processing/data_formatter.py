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
import matplotlib.pyplot as plt
import utils

COLUMNS_TO_DROP = ['open', 'dividends', 'stock splits']

class DataProcessor:
    def __init__(self,  data: pd.DataFrame,
                 identifier_column: str = 'ticker', volatility_window: int = training.config.VOLATILITY_WINDOW):
        self.data = data.reset_index().rename(columns={'index': 'idx'})
        self.identifier = identifier_column
        self.volatility_window = volatility_window
        
    def format_column_names(self):
        self.data.columns = self.data.columns.str.lower()

    def get_date_columns(self):
        self.data['date'] = pd.to_datetime(self.data['date'], utc=True)
        self.data['date'] = self.data['date'].dt.tz_localize(None)
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_the_week'] = self.data['date'].dt.dayofweek
        self.data['day_of_the_month'] = self.data['date'].dt.day
        self.data['year'] = self.data['date'].dt.year
        self.data['weekday_count'] = self.data.groupby(['ticker', 'month', 'year', 'day_of_the_week'])['day_of_the_week'].cumcount()

    def fix_idx_column(self, starting_index: int = 0):
        self.data = self.data.sort_values(['ticker', 'date'])
        self.data['idx'] = self.data.groupby('ticker')['idx'].cumcount() + starting_index

    def get_log_volume_change(self, volume: str = 'volume'):
        self.data['log_volume'] = np.log(self.data[volume])
        self.data['log_volume_change'] = self.data.groupby(self.identifier)['log_volume'].diff()

    def get_log_returns(self, close_price: str = 'close'):
        self.data['log_close'] = np.log(self.data[close_price])
        self.data['log_returns'] = self.data.groupby(self.identifier)['log_close'].diff()
        self.data['log_returns_absolute'] = self.data['log_returns'].abs()
        self.data['log_returns_squared'] = self.data['log_returns']**2

    def get_historical_volatility(self):
        if 'log_returns' in self.data.columns:
            self.data[f'historical_volatility_5_day'] = self.data.groupby(self.identifier)['log_returns'].rolling(self.volatility_window).mean().reset_index(0, drop=True)
        else:
            raise ValueError('Must create log returns before calculating volatility!')

    def get_target_volatility(self):
        '''AVOIDS DATA LEAKAGE
        For example, at time t, the target volatility will be standard deviation for t+1 to t+VOLATILITY_WINDOW
        While historical volatility will represent volatility between t-VOLATILITY_WINDOW and t'''

        if 'log_returns' in self.data.columns:
            self.data['volatility_target'] = self.data.groupby(self.identifier)['log_returns'].shift(-training.config.VOLATILITY_WINDOW).rolling(training.config.VOLATILITY_WINDOW).std().reset_index(0, drop=True)
            self.data['volatility'] = np.sqrt(self.data.groupby(self.identifier)['log_returns_squared'].shift(-training.config.VOLATILITY_WINDOW).rolling(training.config.VOLATILITY_WINDOW).mean().reset_index(0, drop=True))
        else:
            raise ValueError('Must create log returns before calculating volatility!')

    def get_previous_day_range(self, high_price: str = 'high', low_price: str = 'low'):
        if high_price and low_price in self.data.columns:
            self.data['log_prev_day_range'] = np.log(self.data.groupby(self.identifier)[high_price].shift(1)) - \
                np.log(self.data.groupby(self.identifier)[low_price].shift(1))

    def fill_blank_target_rows(self):
        '''Fills in rows where there is no target volatility with 0
        This method is only applicable to new data predictions. I can fill null target values with 0 since they will not be used in the encoder'''
        self.data[['volatility_target', 'volatility']] = self.data[['volatility_target', 'volatility']].fillna(0)

    def drop_unused_columns(self, columns_to_drop: List[str]):
        self.data.drop(columns=columns_to_drop, inplace=True)

    def drop_blank_rows(self):
        self.data.dropna(axis=0, inplace=True)
    
    def add_earnings_periods(self):
        self.data['earnings_period_new'] = self.data.apply(lambda row: 1.0 if row['day_of_the_month'] in np.arange(10, 22, 1) \
                                                       and row['month'] in [1, 4, 7, 10] and row['day_of_the_week'] != 4 else 0, axis=1)
    def add_cpi_report(self):
        cpi_data = pd.read_csv(core.CPI_DATA)
        cpi_data['cpi_date'] = pd.to_datetime(cpi_data['cpi_date'])
        self.data['date'] = pd.to_datetime(self.data['date']).dt.date.astype('datetime64[ns]')
        self.data = pd.merge(self.data, cpi_data[['cpi_date']], left_on='date', right_on='cpi_date', how = 'left')
        self.data['cpi_date'] = self.data['cpi_date'].apply(lambda x: 0 if pd.isnull(x) else 1)
        self.data.rename(columns={'cpi_date': 'cpi_report'}, inplace=True)

    def positive_or_negative_returns(self):
        self.data['positive_or_negative'] = self.data['log_returns'].apply(lambda row: 1 if row > 0 else 0)

    def quadruple_witching(self):
        self.data['quadruple_witching'] = self.data.apply(lambda row: 1 if row['day_of_the_week'] == 4 and \
                                                          row['month'] in ([3, 6, 9, 12]) and row['weekday_count'] == 3 else 0, axis=1)
    def monthly_options_expiration(self):
        self.data['monthly_options_expiration'] = self.data.apply(lambda row: 1 if row['day_of_the_week'] == 4 and row['weekday_count'] == 3 else 0, axis=1)

    def add_election_date_ranges(self):
        election_dates = utils.generate_election_date_ranges()
        self.data['election_dates'] = self.data.apply(lambda row: 1 if row['date'].strftime('%Y-%m-%d') in election_dates else 0, axis=1)

    def employment_report(self):
        self.data['employment_report'] == self.data.apply(lambda row: 1 if row['day_of_the_week'] == 4 and row['weekday_count'] == 1 else 0, axis=1)

    def enrich_rare_features(self, rare_features: List[str] = training.config.FEATURES_TO_ENRICH):
        for feature in rare_features:
            self.data[f'{feature}_new'] = self.data.groupby(self.identifier)[feature].shift(-training.config.VOLATILITY_WINDOW+1).rolling(training.config.VOLATILITY_WINDOW).max().reset_index(0, drop = True)
            self.data[f'{feature}_new'] = self.data[f'{feature}_new'].fillna(0)



def data_management_pipeline(data, starting_idx_value: int = 0):
    dm = DataProcessor(data)
    dm.format_column_names()
    dm.get_date_columns()
    dm.get_log_returns()
    dm.get_log_volume_change()
    dm.get_historical_volatility()
    dm.get_target_volatility()
    dm.get_previous_day_range()
    dm.drop_unused_columns(COLUMNS_TO_DROP)
    dm.add_earnings_periods()
    dm.add_cpi_report()
    dm.add_election_date_ranges()
    dm.positive_or_negative_returns()
    dm.quadruple_witching()
    dm.monthly_options_expiration()
    dm.enrich_rare_features()
    dm.fill_blank_target_rows()
    dm.drop_blank_rows()
    dm.fix_idx_column(starting_idx_value)
    return dm.data

def persist_column_type(data):
    data[training.config.TIME_VARYING_KNOWN_CATEGORICALS+training.config.TIME_VARYING_UNKNOWN_CATEGORICALS]\
          = data[training.config.TIME_VARYING_KNOWN_CATEGORICALS+training.config.TIME_VARYING_UNKNOWN_CATEGORICALS].astype(str)
    return data

