import pandas as pd
from typing import Union
import datetime as dt
from pydantic import BaseModel, validator, ValidationError
import datetime as dt
from typing import List, Dict, Tuple
import core


class InputData(BaseModel):
    tickers: List[str]
    prediction_start_date: str

    @validator('tickers')
    def validate_tickers(cls, tickers):
        if any(ticker not in core.TICKERS for ticker in tickers):
            raise ValueError('Invalid ticker. Please select a ticker from the list of available tickers.\nAvailable tickers are: {}'.format(core.TICKERS))
        return tickers
    
    @validator('prediction_start_date')
    def validate_dates(cls, prediction_start_date):
        try:
            prediction_start_date = dt.datetime.strptime(prediction_start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Invalid date. Please enter a date in the format YYYY-MM-DD')

        market_schedule = pd.read_csv(core.MARKET_SCHEDULE)
        if prediction_start_date.strftime('%Y-%m-%d') not in market_schedule['market_date'].values:
            raise ValueError('Prediction start date must be a market day. Select a different date')
        
        elif prediction_start_date.year != 2023:
            raise ValueError('Prediction start date must be in 2023.')
        
        elif (prediction_start_date - dt.datetime.now()).days > 7:
            raise ValueError('Prediction start date must be within 5 market days of the last market date')

        return prediction_start_date.strftime('%Y-%m-%d')

def validate_inputs(input_data: Dict[List[str], str]):
    errors = None
    try:
        validated_data = InputData(**input_data)
    except ValidationError as error:
        errors = error.json()
    return input_data, errors

