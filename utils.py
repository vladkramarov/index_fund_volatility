import pandas as pd
import core 
import training.config
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pytorch_forecasting import TemporalFusionTransformer

def split_train_valid_test(data:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_idx = data['idx'].max()
    train_end_idx = max_idx - training.config.VALIDATION_DATA_LENGTH - training.config.TEST_DATA_LENGTH
    valid_end_idx = max_idx - training.config.TEST_DATA_LENGTH
    train = data[data['idx'] <= train_end_idx]
    valid = data[(data['idx'] > train_end_idx) & (data['idx'] <= valid_end_idx)]
    test = data[data['idx'] > valid_end_idx]
    return train, valid, test

def save_train_valid_test(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame):
    train_data.to_csv(core.TRAIN_DATA_PATH, index = False)
    valid_data.to_csv(core.VALID_DATA_PATH, index = False)
    test_data.to_csv(core.TEST_DATA_PATH, index = False)

def save_model_interpretation_images(model_intepretation: Dict):

    for parameter in model_intepretation.keys():
        model_intepretation[parameter].savefig(core.IMAGE_PATH / f'{parameter}.png')
    




