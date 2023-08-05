import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import core
import statsmodels.api as sm
import training.config
import loader
from typing import List, Union, Tuple
from scipy.stats import probplot, skew, kurtosis
from statsmodels.tsa.stattools import adfuller

train_data = loader.get_training_data()
def compare_train_valid_test_splits(ticker: str):
    train = loader.get_training_data()
    valid = loader.get_valid_data()
    test = loader.get_test_data()
    fig, ax = plt.subplots(1,1, figsize=(20, 5))
    sns.kdeplot(data = train.query(f'ticker == "{ticker}"')[training.config.TARGET], ax=ax, label='train', fill=True, alpha=0.2, color='blue')
    sns.kdeplot(data = valid.query(f'ticker == "{ticker}"')[training.config.TARGET], ax=ax, label='valid', fill=True, alpha=0.2, color='red')
    sns.kdeplot(data = test.query(f'ticker == "{ticker}"')[training.config.TARGET], ax=ax, label='test', fill=True, alpha=0.2, color='green')
    ax.legend()
    plt.show()

def get_distribution_statistics(data: pd.DataFrame = train_data, ticker: str = 'AAPL') ->Tuple[List[float], str]:
    filtered_data = data.query(f'ticker == "{ticker}"')[training.config.TARGET]
    mean = filtered_data.mean()
    var = filtered_data.var()
    skew = filtered_data.skew()
    kurt = filtered_data.kurtosis()
    text = f'{ticker}: Mean: {mean:.3f}\nStd: {var**0.5:.3f}\nSkew: {skew:.3f}\nKurtosis: {kurt:.3f}'
    return [mean, var, skew, kurt], text

def get_adf(data: pd.DataFrame = train_data, ticker: str = 'AAPL'):
    return adfuller(data.query(f'ticker == "{ticker}"')[training.config.TARGET])
    
def qq_plots(data: pd.DataFrame = train_data, tickers: List[str] = core.TICKERS):
    fig, ax = plt.subplots(4, 2, figsize=(15, 9))
    
    for plot, ticker in zip(ax.flat, tickers):
        moments, text = get_distribution_statistics(data, ticker)
        probplot(data.query(f'ticker == "{ticker}"')[training.config.TARGET], plot=plot)
        plot.text(0.05, 0.95, text, transform=plot.transAxes, fontsize=9, verticalalignment='top')
        #bold f string
        plot.set_title(f'Probability Plot for {ticker}')
    fig.tight_layout()
    plt.show()


def plot_autocorrelations(data: pd.DataFrame = train_data, tickers: List[str] = core.TICKERS, lags: int = 50):
    fig, ax = plt.subplots(len(tickers), 2, figsize=(12, 12))
    for ax_row, ticker in zip(ax, tickers):
        ax1, ax2 = ax_row
        sm.graphics.tsa.plot_acf(data.query(f'ticker == "{ticker}"')[training.config.TARGET], lags=lags, ax=ax1, zero = False)
        sm.graphics.tsa.plot_pacf(data.query(f'ticker == "{ticker}"')[training.config.TARGET], lags=lags, ax=ax2, zero = False)
        ax1.set_title(f'ACF for {ticker}')
        ax1.set_ylim(-0.1, 0.5)
        ax2.set_title(f'PACF for {ticker}')
        ax2.set_ylim(-0.1, 0.5)
        ax2.text(0.05, 0.95, f'ADF: {get_adf(data, ticker)[1]:.3e}', transform=ax2.transAxes, fontsize=9, verticalalignment='top')
    fig.tight_layout()
    plt.show()
        

def violin_plots(data: pd.DataFrame = train_data, hue_feature: str = 'ticker'):
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))

    if hue_feature == 'ticker':
        sns.violinplot(data=data, y='ticker', x=training.config.TARGET, ax=ax, orient='h')
    elif hue_feature == 'month' or hue_feature == 'day_of_week':
        data_by_date = data.groupby('date')[[training.config.TARGET, hue_feature]].mean().reset_index() 
        sns.violinplot(data=data_by_date, y=hue_feature, x=training.config.TARGET, ax=ax, orient='h')
    else:
        raise ValueError(f'Invalid grouping feature: {hue_feature}')
    
    ax.set_title(f'Violin plot of {training.config.TARGET} by {hue_feature}')
    plt.show()




