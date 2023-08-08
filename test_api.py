import requests
import pandas as pd
import data_processing.new_data_processor as new_data_processor
import loader
from pytorch_forecasting import TimeSeriesDataSet
import data_processing.output_formatter as output_formatter
local_host = 'http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com'
#''http://127.0.0.1:8000'
#''
# input_data = {'tickers': ['AAPL'], 'prediction_start_date': '2023-07-25'}

new_data, _ = new_data_processor.new_data_pipeline(tickers = ['AAPL'], prediction_start_date = '2023-07-25')
new_data['date'] = new_data['date'].astype(str)
data = {'processed_data': new_data.to_dict(orient='records')}

results = requests.post(local_host + '/predict_new', json=data)
print(results.content)

ts_dataset_params = loader.get_timeseries_params()
ts_dataset = TimeSeriesDataSet.from_parameters(ts_dataset_params, new_data, predict=False)
model = loader.get_model()
preds = model.predict(ts_dataset, return_index=True, return_x=True, mode='quantiles')
output = output_formatter.OutputFormatter(earliest_prediction_date='2023-07-25', prediction=preds)
formatted = output.get_unaligned_results(new_data)
str(pd.DataFrame(preds[-1]).to_dict(orient='records'))
q = results.json()
dict(q['results'])

q = preds[0]
q.shape
final = preds[-1]
quant = [0.1, 0.5, 0.9]

q[:, :, ].shape
for position, quantile in enumerate(quant):
    quantile_preds = preds[:, :, position]