# 1. Overview

In this project, I use Temporal Fusion Transformer (TFT) to forecast 10-day volatility of 7 sector-specific ETFs:

 - XLK - Technology
 - XLP - Consumer Staples
 - XLF - Financial
 - XLV - Health Care
 - XLE - Energy
 - XLI - Industrial
 - XLU - Utilities

Given the limited availability of high-frequency data (e.g. 5-min data), I opted to forecast **5-day Historical Volatility** which is defined as the standard deviation of squared log daily returns across 5 days. More concretely, given the inputs X<sub>t-n</sub> to X<sub>t</sub>, the model predicts Y<sub>t+1</sub> to Y<sub>t+10</sub>. 

One of the key features of TFT architecture is the ability to output prediction *intervals*, by using Quantile Loss function. The model was trained to provide 3 quantile results - P10, P50, and P90. Another important feature of TFT is the ability to output multi-horizon forecast. For this project, the model was trained to predict 10-day ahead volatility.

It must be noted that, although, volatility is one of the only stock market indicators that is somewhat autocorrelated, it is still challenging to forecast it for a few reasons:
- Volatility can be affected by a plethora of factors, some of which are hard to foresee
- It has no apparent seasonality


The model can be accessed via an API deployed on AWS Elastic Beanstalk using the following [link](http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com/predict). Due to the size of the model and the necessary imports, the API is deployed on t2.medium instance.

 ## 2. Data Source

 All daily stock data was obtained from Yahoo Finance.

 ## 3. Training

Run [training](training/training.py) script to train the model. 
Bayesian optimization was conducted to select optimal hyperparameters for the model. Values for key hyperparameters can be found in the [config](training/config.py) 

## 4. Results

- ![1 day ahead predictions](img/results_over_time_1_days_ahead.png?raw=True "1-day Ahead Forecasts")
- ![3 day ahead predictions](img/results_over_time_3_days_ahead.png?raw=True "3-day Ahead Forecasts") 
- ![7 day ahead predictions](img/results_over_time_7_days_ahead.png?raw=True "7-day Ahead Forecasts") 
- ![10 day ahead predictions](img/results_over_time_10_days_ahead.png?raw=True "10-day Ahead Forecasts") 

## 5. Interpretability

One of the distinguishing aspects of TFT, beyond its forecasting abilities, is its emphasis on interpretability. This ensures that users can not only get accurate forecasts but also insights into why certain predictions are made. More concretely, the architecture of TFT allows to intepret variable importance (in both Encoder and Decoder) as well as show which time steps were most relevant when making a forecast.

- ![Encoder variable importance](img/encoder_variables.png?raw=True "Encoder Variables Importance")
- ![Decoder Variable Importance](img/decoder_variables.png?raw=True "Decoder Variables Importance")
- ![Temporal Distribution of Attention](img/attention.png?raw=True "Temporal Distribution of Attention")

