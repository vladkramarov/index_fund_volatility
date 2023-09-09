import requests
import pandas as pd


HOSTS = {'aws': 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com/predict',
         'local': 'http://localhost:8000/predict'}

def test_api(host_name: str = 'aws', tickers: list = ['XLK'], prediction_start_date: str = '2023-07-25'):
    host = HOSTS[host_name]
    input_data = {'tickers': tickers, 'prediction_start_date': prediction_start_date}

    return requests.post(host, json=input_data)

if __name__ == "__main__":
    response = test_api(host_name='aws')

response.content