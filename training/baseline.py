import numpy as np
import pandas as pd
from pytorch_forecasting import Baseline, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, MAPE, RMSE
import data_processing.dataset_and_loaders as dataset_and_loaders
import importlib
import loader
import core
importlib.reload(dataset_and_loaders)

training_dataset, validation_dataset, train_dataloader, val_dataloader = dataset_and_loaders.get_timeseries_datasets_and_dataloaders()

def get_baseline_preds(validation_dataset: TimeSeriesDataSet = validation_dataset):
    base = Baseline()
    baseline_predictions = base.predict(validation_dataset, return_index=True, return_x=True)
    return baseline_predictions

# def get_baseline_smape(baseline_predictions):
#     actual = tsdataset_and_loaders.get_true_val_labels(val_dataloader).to("mps:0")
#     actual.to("mps:0")
#     smape = RMSE()(baseline_predictions, actual)
#     return smape

# if __name__ == "__main__":
#     baseline_predictions = get_baseline_preds()
#     baseline_smape = get_baseline_smape(baseline_predictions)
#     print(baseline_smape.item())

# preds = get_baseline_preds()
# valid = loader.get_valid_data()
# result = preds.index
# result
# result[core.COLUMNS_TO_ADD_TO_RESULTS] = preds.output.to('cpu').squeeze().numpy()
# result = pd.merge(left=result, right=valid[['ticker', 'idx', 'log_returns_squared']], left_on=['ticker', 'idx'], right_on=['ticker', 'idx'])
# result
# result['true_val'] = valid['log_returns_squared']
# q = validation_dataset.data['target'][0].to('cpu').squeeze().numpy()
# qq = validation_dataset.index
# qq
# smape = get_baseline_smape(preds)

# smape
