from fastapi import FastAPI, HTTPException
import uvicorn
import deployment.predict
import data_processing.input_validation as input_validation
from typing import Dict, List
import data_processing.new_data_processor as new_data_processor
import loader
import numpy as np
import pandas as pd
import logging
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import time
import json
import data_processing.output_formatter as output_formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = FastAPI()

@application.get("/")
def read_root():
    return {"Hello": "World"}

@application.post("/predict")
async def predict(input_data: Dict):
    input_data, errors = input_validation.validate_inputs(input_data)
    results = {}
    if errors is not None:
        raise HTTPException(status_code=400, detail=str(errors))
    else:
        processed_data, _ = new_data_processor.new_data_pipeline(tickers = input_data['tickers'], prediction_start_date = input_data['prediction_start_date'])
        preds = deployment.predict.predict(processed_data)
        processed_output = deployment.predict.process_output(preds, input_data['prediction_start_date'], processed_data, prediction_data_type = 'torch')
        processed_output.replace(np.nan, "N/A", inplace=True)
        results['results'] = processed_output.to_dict(orient='records')
        # results['results'] = type(preds)
    results['errors'] = errors
    return results