import requests
import pandas as pd

local_host = 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com'
input_data = {'tickers': ['AAPL'], 'prediction_start_date': '2023-07-25'}

results = requests.post(local_host + '/predict', json=input_data)

results.json()