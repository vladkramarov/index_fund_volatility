import numpy as np
import pandas as pd
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from typing import Dict, List
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import src.deployment.prediction_pipeline as prediction_pipeline
import src.deployment.input_validation as input_validation
import src.data_processing.new_data_processor as new_data_processor
import src.loader as loader


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
        processed_data, _ = new_data_processor.new_data_pipeline(
            tickers=input_data["tickers"],
            prediction_start_date=input_data["prediction_start_date"],
        )
        preds = prediction_pipeline.predict(processed_data)
        processed_output = prediction_pipeline.process_output(
            preds,
            input_data["prediction_start_date"],
            processed_data,
            prediction_data_type="torch",
        )
        processed_output.replace(np.nan, "N/A", inplace=True)
        results["results"] = processed_output.to_dict(orient="records")

    results["errors"] = errors
    return results
