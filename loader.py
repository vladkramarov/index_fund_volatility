import yfinance as yf
import pandas as pd
import datetime as dt
from typing import List, Union
import numpy as np
import importlib
import core
from pytorch_forecasting import TemporalFusionTransformer
import joblib

importlib.reload(core)


def get_data_from_api(
    tickers: List[str] = core.TICKERS,
    start_date: Union[dt.datetime, str] = None,
    end_date: Union[dt.datetime, None] = None,
) -> pd.DataFrame:
    df_list = []
    for ticker in tickers:
        ticker_searcher = yf.Ticker(ticker)
        if end_date:
            ticker_data = ticker_searcher.history(start=start_date, end=end_date)
        else:
            ticker_data = ticker_searcher.history(start=start_date)
        ticker_data["ticker"] = ticker
        ticker_data.reset_index(inplace=True)
        df_list.append(ticker_data)

    return pd.concat(df_list, axis=0)


def get_earning_dates(tickers: List[str] = core.TICKERS):
    earning_dates_list = []
    for ticker in tickers:
        earning_dates = yf.Ticker(ticker).get_earnings_dates(85)
        earning_dates["ticker"] = ticker
        earning_dates.reset_index(inplace=True)
        earning_dates_list.append(earning_dates)
    return pd.concat(earning_dates_list, axis=0)


def get_data_from_csv(path: str):
    return pd.read_csv(path)


def get_training_data():
    return pd.read_csv(core.TRAIN_DATA_PATH)


def get_valid_data():
    return pd.read_csv(core.VALID_DATA_PATH)


def get_test_data():
    return pd.read_csv(core.TEST_DATA_PATH)


def get_market_schedule():
    market_schedule = pd.read_csv(core.MARKET_SCHEDULE)
    market_schedule["market_date"] = pd.to_datetime(market_schedule["market_date"])
    market_schedule.sort_values(by="market_date", ascending=True, inplace=True)
    return market_schedule


def get_data_for_training(concat_data: bool = True):
    train = get_training_data()
    valid = get_valid_data()
    if concat_data:
        return pd.concat([train, valid], axis=0, ignore_index=True)
    else:
        return train, valid


def get_model():
    return TemporalFusionTransformer.load_from_checkpoint(core.BEST_MODEL_PATH)


def get_timeseries_params():
    return joblib.load(core.TIMESERIES_DATASET_PARAMS)
