import requests
import pandas as pd

local_host = 'http://127.0.0.1:8000'
input_data = {'tickers': ['AAPL'], 'prediction_start_date': 123}

results = requests.post(local_host + '/predict', json=input_data)