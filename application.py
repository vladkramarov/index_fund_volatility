from fastapi import FastAPI, HTTPException
import deployment.predict 
import uvicorn
import data_processing.input_validation as input_validation
from typing import Dict, List
import data_processing.new_data_processor as new_data_processor
import loader
import numpy as np
application = FastAPI()

@application.get("/")
def read_root():
    return {"Hello": "World"}

@application.post("/predict")
async def predict(input_data: Dict):
    input_data, errors = input_validation.validate_inputs(input_data)
    results = {}
    if errors is not None:
        raise HTTPException(status_code=400, detail=errors)
    else:
        processed_data, _ = new_data_processor.new_data_pipeline(tickers = input_data['tickers'], prediction_start_date = input_data['prediction_start_date'])
        model = loader.get_model()
        ts_dataset_params = loader.get_timeseries_params()
        ts_dataset = deployment.predict.TimeSeriesDataSet.from_parameters(ts_dataset_params, processed_data, predict=False)
        preds = model.predict(ts_dataset, return_index=True, return_x=True, mode='quantiles')
        processed_output = deployment.predict.process_output(preds, input_data['prediction_start_date'], processed_data)
        processed_output.replace(np.nan, "N/A", inplace=True)
        results['results'] = processed_output.to_dict(orient='records')
    
    results['errors'] = errors
    return results

