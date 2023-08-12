
NUM_WORKERS = 10
MAX_ENCODER_LENGTH = 63
MAX_PREDICTION_LENGTH = 10
BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
LEARNING_RATE =  0.0000069843 
HIDDEN_SIZE = 64 #48 #36 #48
QUANTILES = [0.1, 0.5, 0.9]
HIDDEN_CONTINUOUS_SIZE = 56 #29 #44 
ATTENTION_HEADS = 4 #5 #4

GRADIENT_CLIP_VAL = 0.055219
DROPOUT =  0.27335 

VALIDATION_DATA_LENGTH = 252
TEST_DATA_LENGTH = 252

VOLATILITY_WINDOW = 5


# COLUMNS FOR TIMESERIESDATASET
GROUP_IDS=["ticker"]
TARGET = 'volatility'
STATIC_CATEGORICALS=["ticker"]
TIME_VARYING_KNOWN_CATEGORICALS= ["day_of_the_week", 'month', 'day_of_the_month', 'earnings_date', 'post_earnings_date', 'cpi_report',\
                                  'monthly_options_expiration', 'quadruple_witching'] 
TIME_VARYING_UNKNOWN_CATEGORICALS = ['positive_or_negative']
TIME_VARYING_KNOWN_REALS = ['idx', 'year']
TIME_VARYING_UNKNOWN_REALS=['log_returns_absolute', "log_prev_day_range", "log_volume_change", 'historical_volatility_5_day']


LAST_TEST_DATASET_DATE = '2023-07-20'
LAST_TEST_IDX = 4756
