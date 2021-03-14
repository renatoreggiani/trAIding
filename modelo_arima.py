# -*- coding: utf-8 -*-

# modelos = ['modelo_arima', 'modelo_lstm',  ]
# tickers = ['babs']

'''
# atualiza tudo
for model in modelos:
    for ticker in tickers:
        m = eval(model)
        m = m(df, ticker)
        m.atualiza(daysfor=2)        
        
# validador tudo
for model in modelos:
    for ticker in tickers:
        m = eval(model)
        m = m(df, ticker)
        validador(m)

x = validador.get_best('babs')
x = 'modelo_lstm'
m = eval(x)
m = m(df, ticker)
m.predict
'''

# In[8]:


import interfaces
import pandas as pd
import numpy as np
import os
import pickle
import json
import datetime

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from functions import stationary_test

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

# In[]:
    
class modelo_arima(AbstractModelo):
    
    def __init__(self, df,  ticker, DaysForUpdate=0):
        self.ticker = ticker
        self.__nome_modelo = self.ticker
        self.df = df
        self.modelo = None
    
    @property
    def nome_modelo(self):
        return f'ARIMA_{self.__nome_modelo}'
    
    def ajusta_dados(self):
        #exclui todas as colunas exceto data e y
        self.df.dropna(subset=['y'], inplace=True)
        for col in self.df.columns: 
            if col not in ['date','y']:
                self.df.drop(col,axis='columns',inplace=True)

    def fit(self)->None:
        is_stat = stationary_test(self.df)
        self.modelo = auto_arima(self.df, stationary=is_stat, start_p=0, start_d=0, start_q=0, max_p=10, max_d=10, max_q=10, 
                             start_P=0, start_D=0, start_Q=0, max_P=15, max_D=15, max_Q=15, m=15, seasonal='False', n_fits=50,
                             error_action='warn', trace=True, suppress_warnings=True, stepwise=False, random_state=20, n_jobs=-1)

    def salva_modelo(self):
        try:
            with open('modelos/arima.json','r') as jfile:
                jdata = json.load(jfile)
        except:
            jdata = {}
        hoje = datetime.datetime.now().strftime("%Y-%m-%d")
        jdata[self.ticker].update({"parametros":self.modelo.order + self.modelo.seasonal_order,"train_date":hoje})
        json.dump(jdata, open('modelos/arima.json','w'), indent=4)
        pickle.dump(self.modelo, open("modelos/arima/"+self.ticker,"wb"))

    def forecast(self)->float:
        pass

    def atualiza_modelo(self):
        pass
    
    def carrega_modelo(self):
        pass