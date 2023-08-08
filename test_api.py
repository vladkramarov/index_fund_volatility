import requests
import pandas as pd
import data_processing.new_data_processor as new_data_processor

local_host = 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com'
#''http://127.0.0.1:8000'
#''
# input_data = {'tickers': ['AAPL'], 'prediction_start_date': '2023-07-25'}

new_data, _ = new_data_processor.new_data_pipeline(tickers = ['AAPL'], prediction_start_date = '2023-07-25')
data = {'processed_data': new_data.to_dict(orient='records')}

results = requests.post(local_host + '/predict_new', json=data)

print(results.json())