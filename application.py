from fastapi import FastAPI, HTTPException
import uvicorn
import data_processing.input_validation as input_validation
from typing import Dict, List
import data_processing.new_data_processor as new_data_processor
import loader
import numpy as np
import pandas as pd
import logging
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import time
import data_processing.output_formatter as output_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = FastAPI()

@application.get("/")
def read_root():
    return {"Hello": "World"}

# @application.post("/predict")
# async def predict(input_data: Dict):
#     input_data, errors = input_validation.validate_inputs(input_data)
#     results = {}
#     if errors is not None:
#         raise HTTPException(status_code=400, detail=errors)
#     else:
#         processed_data, _ = new_data_processor.new_data_pipeline(tickers = input_data['tickers'], prediction_start_date = input_data['prediction_start_date'])
#         logger.info(f'Processed data: {processed_data.head()}')
#         model = loader.get_model()
#         ts_dataset_params = loader.get_timeseries_params()
#         ts_dataset = TimeSeriesDataSet.from_parameters(ts_dataset_params, processed_data, predict=False)
#         start_time = time.time()
#         preds = None
#         try:
#             preds = model.predict(ts_dataset, return_index=True, return_x=True, mode='quantiles')
#         except Exception as e:
#             logger.exception("Error during prediction")

#         end_time = time.time()
#         output = preds[-1].replace(np.nan, "N/A", inplace=True).to_dict()
#         logger.info(f"Model prediction took {end_time - start_time} seconds")
#         # processed_output = deployment.predict.process_output(preds, input_data['prediction_start_date'], processed_data)
#         # processed_output.replace(np.nan, "N/A", inplace=True)
#         results['results'] = output
    
#     results['errors'] = errors
#     return results

@application.post("/predict_new")
async def predict_new(input_data: Dict):
    processed_data = pd.DataFrame(input_data['processed_data'])
    model = loader.get_model()
    ts_dataset_params = loader.get_timeseries_params()
    ts_dataset = TimeSeriesDataSet.from_parameters(ts_dataset_params, processed_data, predict=False)
    results = {}
    preds = model.predict(ts_dataset, return_index=True, return_x=True, mode='quantiles')
    output = preds[0].to('cpu').detach().numpy()
    logger.info(f'Output shape: {output.shape}')
    logger.info(f'Output type: {type(output)}')
    output = output.reshape(output.shape[0], -1)
    final = preds.index
    cols = [f'q_{i}' for i in range(1, output.shape[1] + 1)]
    final[cols] = output
    final.replace(np.nan, "N/A", inplace=True)
    results['results'] = final.to_dict(orient='records')
    return results