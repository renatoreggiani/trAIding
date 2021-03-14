# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:19:05 2021

@author: F8564619
"""
# !conda install libpython m2w64-toolchain -c msys2
# !conda install pystan -c conda-forge
# !pip install fbprophet

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py

from interfaces import AbstractModelo

#%%


class ModeloProphet(AbstractModelo):
    
    def __init__(self, df, ticker, periods=60):
        self.__nome_modelo = ticker
        self.df_in = df.copy()
        self.periods = periods
        self.cap = 0
        self.df_prophet = self.ajusta_dados()
        self.model = self.model()
        self.forecast = self.p_predict()
        
    @property
    def nome_modelo(self)->str:
        pass 
    
    @property
    def cap(self):
        cap = int(self.df_in[self.col_y].max() * 1.1)
        return cap
    
    @cap.setter
    def cap(self, value):
        if value <= 0:
            self._cap = 2
        else:
            self._cap = value
        
   
    def ajusta_dados(self):
        df_prophet = self.df_in.copy()
        
        df_prophet['floor'] = 0
        df_prophet['cap'] = self.cap
        df_prophet = df_prophet[['ds', 'y','floor','cap']].sort_values('ds')
        return df_prophet.dropna()
    
    def model(self):
        m = Prophet(growth='logistic',daily_seasonality=False, yearly_seasonality=True, n_changepoints=40,
                    weekly_seasonality=True, seasonality_mode='additive',
                   seasonality_prior_scale=100)
        m.add_country_holidays(country_name='BR')
        return m.fit(self.df_prophet)
        
    def p_predict(self):
        future = self.model.make_future_dataframe(periods=self.periods)
        future['floor'] = 1
        future['cap'] = self.cap
        return self.model.predict(future)
    
    def plot(self):
        
        fig = plot_plotly(self.model, self.forecast) 
        py.iplot(fig)
        
    def plot_sazonalidade(self):
        self.model.plot_components(self.forecast);
        




    

    
    @abstractmethod
    def fit(self)-> None:
        pass

    @abstractmethod
    def forecast(self)-> float:
        pass
    
    @abstractmethod
    def carrega_modelo(self):
        pass

    @abstractmethod
    def salva_modelo(self):
        pass
    
    @abstractmethod
    def atualiza_modelo(self):
        pass