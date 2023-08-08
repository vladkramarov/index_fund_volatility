import requests
import pandas as pd
import data_processing.new_data_processor as new_data_processor

local_host = 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com'
#''http://127.0.0.1:8000'
#''
input_data = {'tickers': ['AAPL'], 'prediction_start_date': '2023-07-25'}

# new_data = new_data_processor.new_data_pipeline(tickers = 'AAPL', prediction_start_date = '2023-08-07')
# new_data
# data = {'processed_data': new_data}
results = requests.post(local_host + '/predict', json=input_data)

print(results.json())