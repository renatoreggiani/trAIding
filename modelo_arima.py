'''
# atualiza tudo
for model in modelos:
    for ticker in tickers:
        m = eval(model)
        m = m(df, ticker)
        m.atualiza(daysfor=2)
        m.salva_modelo

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


import os
import pickle
import json
import datetime

from interfaces import AbstractModelo
from pmdarima.arima import auto_arima
from functions import stationary_test

import warnings
warnings.filterwarnings('ignore')

# In[]:


class modelo_arima(AbstractModelo):

    def __init__(self, df,  ticker, days_force_update=0):
        self.ticker = ticker
        self.__nome_modelo = self.ticker
        self.df = df
        self.modelo = None

    @property
    def nome_modelo(self):
        return f'ARIMA_{self.__nome_modelo}'

    def ajusta_dados(self):
        # exclui todas as colunas exceto data e y
        self.df.dropna(subset=['y'], inplace=True)
        for col in self.df.columns:
            if col not in ['y']:
                self.df.drop(col, axis='columns', inplace=True)

    def fit(self) -> None:
        is_stat = stationary_test(self.df['y'])
        arima_model = auto_arima(self.df['y'], stationary=is_stat, start_p=0, start_d=0, start_q=0,
                                 max_p=10, max_d=10, max_q=10, start_P=0, start_D=0, start_Q=0,
                                 max_P=15, max_D=15, max_Q=15, m=15, seasonal='False', n_fits=50,
                                 error_action='warn', trace=True, suppress_warnings=True,
                                 stepwise=False, random_state=20, n_jobs=-1)
        self.modelo = arima_model.fit(y=self.df['y'])

    def salva_modelo(self):
        try:
            with open('modelos/arima/arima.json', 'r') as jfile:
                jdata = json.load(jfile)
        except FileNotFoundError:
            jdata = {}
        hoje = datetime.datetime.now().strftime("%Y-%m-%d")
        if self.ticker not in jdata:
            jdata[self.ticker] = {}
        jdata[self.ticker]['parametros'] = self.modelo.order + self.modelo.seasonal_order
        jdata[self.ticker]['train_date'] = hoje
        json.dump(jdata, open('modelos/arima/arima.json', 'w'), indent=4)
        pickle.dump(self.modelo, open("modelos/arima/arima_"+self.ticker, "wb"))

    def forecast(self) -> float:
        self.modelo.predict(n_periods=1, alpha=0.05)[0]

    def atualiza_modelo(self):
        train_date = datetime.datetime.strptime(os.path.getmtime("modelos/arima/arima." +
                                                                 self.ticker+".bin"))
        if ((datetime.datetime.now() - train_date).days) > self.days_force_update:
            self.fit()

    def carrega_modelo(self):
        self.modelo = pickle.load(open("modelos/arima/arima."+self.ticker+".bin", 'rb'))


# In[]

import yfinance as yf
ticker = 'BBAS3.SA'
tkr = yf.Ticker(ticker)
df = tkr.history(period='1y', interval='1d')
df = df.rename(columns={'Close': 'y'}, inplace=False)

m = modelo_arima(df, ticker)
m.ajusta_dados()
m.fit()
m.salva_modelo()
f = m.forecast()
