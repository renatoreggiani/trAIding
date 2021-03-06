# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:28:55 2021

@author: Renato
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyfolio as pf
import warnings
warnings.filterwarnings('ignore')

import pandas_datareader as pdr
import pandas_datareader.data as web

import arima_coach
from pmdarima import auto_arima
#%%



df_train = web.DataReader('ITUB4.SA', 'yahoo', start='2015-01-01', end='2017-12-31')
s = df_train.Close

df_teste = web.DataReader('ITUB4.SA', 'yahoo', start='2015-01-01', end='2020-02-22')
s = df_teste.Close


arima_coach.get_arima_model(s)



auto_arima(df_train.Close, stationary=False, start_p=0, start_d=0, start_q=0, max_p=10, max_d=10, max_q=10, 
           seasonal=False, error_action='warn', trace=True, suppress_warnings=True, stepwise=False, random_state=20, 
           information_criterion='aic', n_fits=50, n_jobs=-1)

pf.create_full_tear_sheet(carteira["retorno"], benchmark_rets=retorno["^BVSP"])


fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_rolling_beta(carteira["retorno"], factor_returns=retorno["^BVSP"], ax=ax1)
plt.ylim((0.8, 1.4));


s.describe()
