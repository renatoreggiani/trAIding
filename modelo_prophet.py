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

    def __init__(self, df, ticker, periods=1):
        self.modelo = None
        self.ticker = ticker
        self.__nome_modelo = self.nome_modelo
        self.df = df.copy()
        self.periods = periods
        self.df_prophet = self.ajusta_dados()
        # self.model = self.fit()

        # self.fit()
        # self.forecast = self.forecast()

    @property
    def nome_modelo(self) -> str:
        return f'modelo_prophet_{self.ticker}'

    def ajusta_dados(self):
        # df_prophet = self.df.copy()
        # df_prophet = df_prophet[['ds', 'y']].sort_values('ds')
        return self.df.dropna()

    def fit(self):
        self.modelo = Prophet(growth='linear', daily_seasonality=False,
                              yearly_seasonality='auto', n_changepoints=140,
                              weekly_seasonality='auto', seasonality_mode='multiplicative',
                              seasonality_prior_scale=100)
        self.modelo.fit(self.df_prophet)

    def forecast(self, periods=1, plot=False):
        self.future = self.modelo.make_future_dataframe(periods=periods, freq='B')
        if plot:
            return self.modelo.predict(self.future)
        else:
            return self.modelo.predict(self.future)['yhat'].values[-1]

    def plot(self):
        fig = plot_plotly(self.modelo, self.forecast(periods=90, plot=True))
        py.plot(fig)

    def plot_sazonalidade(self):
        self.modelo.plot_components(self.forecast(periods=90, plot=True))

    def carrega_modelo(self):
        pass

    def salva_modelo(self):
        pass

    def atualiza_modelo(self):
        pass


# %% teste
ticker = 'ITUB4.SA'
df = get_finance_data(ticker, period='1y')
# df.y = df.y.pct_change()
m = ModeloProphet(df, ticker)
m.fit()
m.plot()
m.plot_sazonalidade()
m.forecast()

