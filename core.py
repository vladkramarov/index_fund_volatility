from pathlib import Path

ROOT_FOLDER = Path('STOCK_VOLATILITY_NEW').resolve().parent

DATASET_DIR = ROOT_FOLDER / 'data'

ALL_DATA_DIR = DATASET_DIR / 'all_data.csv'
ALL_DATA_NEW_DIR = DATASET_DIR / 'all_data_new.csv'
UNPROCESSED_DATA = DATASET_DIR / 'index_funds_data.csv'
FORMATTED_DATA = DATASET_DIR / 'formatted_data.csv'
DATA_FOR_ANALYSIS = DATASET_DIR / 'new_formatted_data.csv'
EARNINGS_DATA = DATASET_DIR / 'earnings.csv'
CPI_DATA = DATASET_DIR / 'cpi.csv'
MARKET_SCHEDULE = DATASET_DIR / 'nasdaq_schedule.csv'
TRAIN_DATA_PATH = DATASET_DIR / 'train.csv'
VALID_DATA_PATH = DATASET_DIR / 'valid.csv'
TEST_DATA_PATH = DATASET_DIR / 'test.csv'


TICKERS = ['XLK', 'XLP', 'XLF', 'XLV', 'XLE', 'XLI', 'XLU']

FORECAST_HORIZONS = ['one_days', 'two_days', 'three_days', 'four_days', 'five_days']
ADDITIONAL_OUTPUT_COLS = ['volatility_target', 'date', 'ticker', 'idx']

LIGHTNING_LOGS_DIR = ROOT_FOLDER / 'lightning_logs'
IMAGE_PATH = ROOT_FOLDER / 'img'
MODEL_DATA = ROOT_FOLDER /  'model_data'
BEST_MODEL_PATH = MODEL_DATA / 'checkpoint.ckpt'
TIMESERIES_DATASET_PARAMS = MODEL_DATA / 'ts_dataset_params.joblib'


PRES_ELECTION_DATES = ['2004-11-02', '2008-11-04', '2012-11-06', '2016-11-08', '2020-11-03']
MIDTERM_ELECTION_DATES = ['2006-11-07', '2010-11-02', '2014-11-04', '2018-11-06', '2022-11-08']
ELECTION_DATES = PRES_ELECTION_DATES+MIDTERM_ELECTION_DATES