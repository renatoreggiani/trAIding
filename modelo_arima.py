'''
# atualiza e valida tudo
for model in modelos:
    for ticker in tickers:
        m = eval(model)
        m = m(df, ticker, days_for_update=1)
        m.carrega_modelo()
        validador(m)

x = validador.get_best('babs')
x = 'modelo_lstm'
m = eval(x)
m = m(df, ticker)
m.predict
'''

# In[8]:


import os
import pickle
import json
import mensageria

from interfaces import AbstractModelo
from pmdarima.arima import auto_arima
from functions import stationary_test
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# In[]:


class modelo_arima(AbstractModelo):

    def __init__(self, df,  ticker, days_for_update=0):
        self.ticker = ticker
        self.__nome_modelo = 'ARIMA'
        self.__binfilename = 'modelos/arima_' + self.ticker
        self.df = df
        self.modelo = None
        self.days_for_update = days_for_update
        self.ajusta_dados()

    @property
    def nome_modelo(self):
        return 'ARIMA'

    def ajusta_dados(self):
        # exclui todas as colunas exceto data e y
        self.df.dropna(subset=['y'], inplace=True)
        for col in self.df.columns:
            if col not in ['y']:
                self.df.drop(col, axis='columns', inplace=True)

    def fit(self) -> None:
        mensageria.msg_starting_fitting(self.nome_modelo, self.ticker)
        is_stat = stationary_test(self.df['y'])
        arima_model = auto_arima(self.df['y'], stationary=is_stat, start_p=0, start_d=0, start_q=0,
                                 max_p=10, max_d=10, max_q=10, start_P=0, start_D=0, start_Q=0,
                                 max_P=15, max_D=15, max_Q=15, m=15, seasonal='False', n_fits=50,
                                 error_action='warn', trace=True, suppress_warnings=True,
                                 stepwise=False, random_state=20, n_jobs=-1)
        print('\n')
        # mensageria.msg_fitting_complete(self.nome_modelo, self.ticker)
        self.modelo = arima_model.fit(y=self.df['y'])
        # self.salva_modelo()

    def salva_modelo(self):
        try:
            with open('modelos/arima.json', 'r') as jfile:
                jdata = json.load(jfile)
        except FileNotFoundError:
            jdata = {}
        hoje = datetime.now().strftime("%Y-%m-%d")
        if self.ticker not in jdata:
            jdata[self.ticker] = {}
        jdata[self.ticker]['parametros'] = self.modelo.order + self.modelo.seasonal_order
        jdata[self.ticker]['train_date'] = hoje
        json.dump(jdata, open('modelos/arima.json', 'w'), indent=4)
        mensageria.msg_saving_model(self.nome_modelo, self.ticker, self.__binfilename)
        pickle.dump(self.modelo, open(self.__binfilename, "wb"))

    def forecast(self) -> float:
        return self.modelo.predict(n_periods=1, alpha=0.05)[0]

    def atualiza_modelo(self):
        try:
            train_date = datetime.fromtimestamp(os.path.getmtime(self.__binfilename))
            if ((datetime.now() - train_date).days) > self.days_for_update:
                mensageria.msg_model_out_of_date(self.nome_modelo, self.ticker, train_date)
                self.fit()
            else:
                mensageria.msg_model_up_to_date(self.nome_modelo, self.ticker, train_date)
        except FileNotFoundError:
            mensageria.msg_file_not_found(self.__binfilename)
            self.fit()

    def carrega_modelo(self):
        try:
            self.modelo = pickle.load(open(self.__binfilename, 'rb'))
            mensageria.msg_loading_model(self.nome_modelo, self.ticker, self.__binfilename)
            return True
            # self.atualiza_modelo()
        except FileNotFoundError:
            mensageria.msg_file_not_found(self.__binfilename)
            return False
            # self.fit()


# In[]
'''
import yfinance as yf

ticker = 'BBAS3.SA'
mensageria.msg_loading_finance_data(ticker)
tkr = yf.Ticker(ticker)
df = tkr.history(period='1y', interval='1d')
df = df.rename(columns={'Close': 'y'}, inplace=False)

m = modelo_arima(df, ticker, days_for_update=7)
m.carrega_modelo()
m.atualiza_modelo()
m.salva_modelo()
f = m.forecast()
'''
