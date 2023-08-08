import requests
import pandas as pd

local_host = 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com'
#'http://127.0.0.1:8000'
#'
input_data = {'tickers': ['AAPL'], 'prediction_start_date': '2023-08-07'}

results = requests.post(local_host + '/predict', json=input_data)
