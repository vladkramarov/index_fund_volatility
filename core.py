from pathlib import Path

ROOT_FOLDER = Path('ETHEREUM_VOLATILITY').resolve().parent

DATASET_DIR = ROOT_FOLDER / 'data'

ALL_DATA_DIR = DATASET_DIR / 'all_data.csv'
ALL_DATA_NEW_DIR = DATASET_DIR / 'all_data_new.csv'
UNPROCESSED_DATA = DATASET_DIR / 'unprocessed_data.csv'
FORMATTED_DATA = DATASET_DIR / 'formatted_data.csv'
DATA_FOR_ANALYSIS = DATASET_DIR / 'new_formatted_data.csv'
EARNINGS_DATA = DATASET_DIR / 'earnings.csv'
CPI_DATA = DATASET_DIR / 'cpi.csv'
MARKET_SCHEDULE = DATASET_DIR / 'nasdaq_schedule.csv'


TRAIN_DATA_PATH = DATASET_DIR / 'train.csv'
VALID_DATA_PATH = DATASET_DIR / 'valid.csv'
TEST_DATA_PATH = DATASET_DIR / 'test.csv'


TICKERS = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NFLX', 'NVDA', 'ADBE']

FORECAST_HORIZONS = ['one_days', 'two_days', 'three_days', 'four_days', 'five_days']
ADDITIONAL_OUTPUT_COLS = ['volatility_target', 'date', 'ticker', 'idx']

LIGHTNING_LOGS_DIR = ROOT_FOLDER / 'lightning_logs'
BEST_MODEL_PATH = ROOT_FOLDER /  'model_path/checkpoints/epoch=9-step=4330.ckpt'
# 'version_175/checkpoints/epoch=21-step=9526.ckpt'
#LIGHNING_LOGS_DIR / 'lightning_logs/version_120/checkpoints/epoch=65-step=30426.ckpt'
TIMESERIES_DATASET_PARAMS = ROOT_FOLDER / 'ts_dataset_params.joblib'

