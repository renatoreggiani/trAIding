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


from functions import get_finance_data

# %%


class ModeloProphet(AbstractModelo):

    """ Classe para rodar o modelo Prophet, https://facebook.github.io/prophet/
    precisa receber um dataframe com colunas 'y' e 'ds'
    """

    def __init__(self, df, ticker, periods=60):
        self.model = Prophet(growth='linear', daily_seasonality=False,
                             yearly_seasonality='auto', n_changepoints=140,
                             weekly_seasonality='auto', seasonality_mode='additive',
                             seasonality_prior_scale=100)
        self.ticker = ticker
        self.__nome_modelo = self.nome_modelo
        self.df_in = df.copy()
        self.periods = periods
        self.df_prophet = self.ajusta_dados()
        # self.model = self.fit()

        self.fit()
        self.forecast = self.forecast()

    @property
    def nome_modelo(self) -> str:
        return f'modelo_prophet_{self.ticker}'

    def ajusta_dados(self):
        df_prophet = self.df_in.copy()
        df_prophet = df_prophet[['ds', 'y']].sort_values('ds')
        return df_prophet.dropna()

    def fit(self):
        self.model.fit(self.df_prophet)

    def forecast(self):
        future = self.model.make_future_dataframe(periods=self.periods, freq='B')
        return self.model.predict(future)

    def plot(self):
        fig = plot_plotly(self.model, self.forecast)
        py.plot(fig)

    def plot_sazonalidade(self):
        self.model.plot_components(self.forecast)

    def carrega_modelo(self):
        pass

    def salva_modelo(self):
        pass

    def atualiza_modelo(self):
        pass


# %%
ticker = 'ITUB4.SA'
df = get_finance_data(ticker)
# df.y = df.y.pct_change()
m = ModeloProphet(df, ticker)
# m.fit()
m.plot()

ModeloProphet?

