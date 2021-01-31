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
#%%



df_train = web.DataReader('ITUB4.SA', 'yahoo', start='2015-01-01', end='2017-12-31')
df_train = web.DataReader('ITUB4.SA', 'yahoo', start='2015-01-01', end='2017-12-31')

pf.create_full_tear_sheet(carteira["retorno"], benchmark_rets=retorno["^BVSP"])


fig, ax1 = plt.subplots(figsize=(16,8))
pf.plot_rolling_beta(carteira["retorno"], factor_returns=retorno["^BVSP"], ax=ax1)
plt.ylim((0.8, 1.4));