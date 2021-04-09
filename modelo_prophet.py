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

import pickle
from datetime import datetime
import os


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
        self.__binfilename = 'modelos/PROPHET_' + self.ticker + '.bin'
        self.lag_regressors = [col for col in df.columns if col.startswith('x_')]
        self.df, self.df_regressors = self.ajusta_dados(df)


    @property
    def nome_modelo(self) -> str:
        return f'modelo_prophet_{self.ticker}'

    def ajusta_dados(self, df):
        df_regressors = df[[col for col in df.columns if col.startswith('x_')]]
        df = df.copy()[['y', 'ds']]
        if not df_regressors.empty:
            df = df.merge(df_regressors.shift(1), left_index=True, right_index=True)
            df_regressors = df_regressors.shift(1, freq='B')
            df_regressors.loc[:,'ds'] = df_regressors.index.values
        return df.dropna(), df_regressors

    def fit(self):
        self.modelo = Prophet(growth='linear', daily_seasonality=False,
                              yearly_seasonality='auto', n_changepoints=140,
                              weekly_seasonality='auto', seasonality_mode='multiplicative',
                              seasonality_prior_scale=100)
        for reg in self.lag_regressors:
            self.modelo.add_regressor(reg)
        self.modelo.fit(self.df)

    def forecast(self, periods=1, plot=False):
        self.future = self.modelo.make_future_dataframe(periods=periods, freq='B')
        if not self.df_regressors.empty:
            self.future = self.future.merge(self.df_regressors, left_on='ds', right_on='ds')

        if plot:
            return self.modelo.predict(self.future)
        else:
            return self.modelo.predict(self.future)['yhat'].values[-1]

    def salva_modelo(self):
        pickle.dump(self.modelo, open(self.__binfilename, "wb"))

    def atualiza_modelo(self, days_for_update=0):
        try:
            train_date = datetime.fromtimestamp(os.path.getmtime(self.__binfilename))
            if ((datetime.now() - train_date).days) > days_for_update:
                # mensageria.msg_model_out_of_date(self.nome_modelo, self.ticker, train_date)
                self.fit()
                self.salva_modelo()
            # else:
                # mensageria.msg_model_up_to_date(self.nome_modelo, self.ticker, train_date)
        except FileNotFoundError:
            # mensageria.msg_file_not_found(self.__binfilename)
            self.fit()
            self.salva_modelo()

    def carrega_modelo(self):
        try:
            self.modelo = pickle.load(open(self.__binfilename, 'rb'))
            # mensageria.msg_loading_model(self.nome_modelo, self.ticker, self.__binfilename)
        except FileNotFoundError as e:
            print(e)

    def plot(self):
        fig = plot_plotly(self.modelo, self.forecast(periods=90, plot=True))
        py.plot(fig)

    def plot_sazonalidade(self):
        self.modelo.plot_components(self.forecast(periods=90, plot=True))


# %% teste

if __name__ == '__main__':
    ticker = 'ITUB4.SA'
    df = get_finance_data(ticker, period='10y')
    # df.y = df.y.pct_change()
    df['x_volume'] = df.loc[:,['Volume']].shift(1, freq='B')
    df.merge(df_reg, right_index=True, left_index=True, how='outer')
    m = ModeloProphet(df, ticker)
    m.fit()
    m.plot()
    m.plot_sazonalidade()
    m.forecast()



m = Prophet()
m.add_country_holidays(country_name='BR')
m.fit(df)


import holidays
feriados= holidays.Brazil()
for feriado in feriados['2020-01-01': '2020-12-31'] :
    print(feriado)





import pandas_market_calendars as mcal




print(mcal.get_calendar_names())



mkdays = mcal.exchange_calendar_bmf.BMFExchangeCalendar()
early = mkdays.schedule(start_date=df.index[1].strftime('%Y-%m-%d'), end_date=df.index[-1].strftime('%Y-%m-%d'))
mcal.date_range(early, frequency='1D')
