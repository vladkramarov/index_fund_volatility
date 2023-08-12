# 1. Overview

In this project, I use Temporal Fusion Transformer (TFT) to forecast 10-day volatility of 7 tech stocks:

 - Adobe (ADBE)
 - Alphabet (GOOGL)
 - Amazon (AMZN)
 - Apple (AAPL)
 - Microsoft (MSFT)
 - Netflix (NFLX)
 - Nvidia (NVDA)

 One of the key features of TFT architecture is the ability to output prediction *intervals*, by using Quantile Loss function. The model was trained to provide 3 quantile results - P10, P50, and P90. Another important feature of TFT is to output multi-horizon forecast. For this project, the model was trained to predict 10-day ahead volatility.

 The model can be accessed via an API deployed on AWS Elastic Beanstalk using the following [link](http://pytorch-final-env.eba-zbppsjpu.us-east-1.elasticbeanstalk.com/predict). Due to the size of the model and the necessary imports, the API is deployed on t2.medium instance. So, don't use it too often, or you will make me go broke :(

 ## 2. Data Source

 All daily stock data was obtained from Yahoo Finance.

 ## 3. Training

Run [training](training/training.py) script to train the model. 
Bayesian optimization was conducted to select optimal hyperparameters for the model. Values for key hyperparameters can be found in the [config](training/config.py) 

## 4. Results


