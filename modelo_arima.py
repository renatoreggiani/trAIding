# -*- coding: utf-8 -*-


# In[8]:


import interfaces
import pandas as pd
import numpy as np
import os
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
    
    def __init__(self, df,  ticker):
        self.modelo = 'arima'
        self.ticker = ticker
        self.__nome_modelo = self.ticker
        self.df = df
    
    @property
    def nome_modelo(self):
        return f'ARIMA_{self.__nome_modelo}'
    
    def ajusta_dados(self):
        #exclui todas as colunas exceto data e y
        for col in self.df.columns: 
            if col not in ['date','y']:
                self.df.drop(col,axis='columns',inplace='True')
    
    def fit(self)-> None:
        is_stat = stationary_test(self.df)
        arima_model = auto_arima(self.df, stationary=is_stat, start_p=0, start_d=0, start_q=0, max_p=10, max_d=10, max_q=10, start_P=0, start_D=0, 
                             start_Q=0, max_P=15, max_D=15, max_Q=15, m=15, seasonal='False', error_action='warn', trace=True, 
                             suppress_warnings=True, stepwise=False, random_state=20, n_fits=50, n_jobs=-1)


    def salva_modelo(self):
        # 
        print('salvando')
        try:
            with open('modelos/resumo.json','r') as jfile:
                jdata = json.load(jfile)
        except:
            jdata = {}
        
    def forecast(self) -> float:
        pass

    def atualiza_modelo(self):
        pass
    
    def carrega_modelo(self):
        pass