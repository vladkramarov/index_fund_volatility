from fastapi import FastAPI, HTTPException
import deployment.predict 
import uvicorn
import data_processing.input_validation as input_validation
from typing import Dict, List
import data_processing.new_data_processor as new_data_processor
import loader
import numpy as np

processed, _ = new_data_processor.new_data_pipeline(tickers = ['AAPL', 'MSFT'], prediction_start_date = '2023-08-03')
preds = deployment.predict.predict(processed)



output = deployment.predict.process_output(preds, '2023-08-03', processed)
output.replace(np.nan, "N/A", inplace=True)
