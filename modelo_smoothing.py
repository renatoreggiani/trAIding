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
# import json
import mensageria

from interfaces import AbstractModelo
from darts.models import ExponentialSmoothing
from datetime import datetime
from darts import TimeSeries

import warnings
warnings.filterwarnings('ignore')

# In[]:


class modelo_smoothing(AbstractModelo):

    def __init__(self, df,  ticker):
        self.ticker = ticker
        self.__nome_modelo = 'ExponentialSmoothing'
        self.__binfilename = 'modelos/ExpSoothing_' + self.ticker + '.bin'
        self.df = df
        self.modelo = None
        self.ajusta_dados()

    @property
    def nome_modelo(self):
        return 'ExponentialSmoothing'

    def ajusta_dados(self):
        self.df = TimeSeries.from_dataframe(self.df, None, 'y', 'B', fill_missing_dates=False)

    def fit(self) -> None:
        mensageria.msg_starting_fitting(self.nome_modelo, self.ticker)
        # is_stat = stationary_test(self.df['y']) || devemos fazer teste de estacionaridade?
        self.modelo = ExponentialSmoothing()
        mensageria.msg_fitting_complete(self.nome_modelo, self.ticker)
        self.modelo.fit(self.df['y'])

    def forecast(self) -> float:
        return self.modelo.predict(1)._df

# com o novo interface proposto, dá pra apagar todas essas funções abaixo

    def salva_modelo(self):
        mensageria.msg_saving_model(self.nome_modelo, self.ticker, self.__binfilename)
        pickle.dump(self.modelo, open(self.__binfilename, "wb"))

    def atualiza_modelo(self, days_for_update=0):
        try:
            train_date = datetime.fromtimestamp(os.path.getmtime(self.__binfilename))
            if ((datetime.now() - train_date).days) > days_for_update:
                mensageria.msg_model_out_of_date(self.nome_modelo, self.ticker, train_date)
                self.fit()
                self.salva_modelo()
            else:
                mensageria.msg_model_up_to_date(self.nome_modelo, self.ticker, train_date)
        except FileNotFoundError:
            mensageria.msg_file_not_found(self.__binfilename)
            self.fit()
            self.salva_modelo()

    def carrega_modelo(self):
        try:
            self.modelo = pickle.load(open(self.__binfilename, 'rb'))
            mensageria.msg_loading_model(self.nome_modelo, self.ticker, self.__binfilename)
        except FileNotFoundError:
            mensageria.msg_file_not_found(self.__binfilename)


# In[]

from functions import get_finance_data

# In[]

ticker = 'BBAS3.SA'
mensageria.msg_loading_finance_data(ticker)
df = get_finance_data(ticker, period='5y')

m = modelo_smoothing(df, ticker)
m.fit()
# m.carrega_modelo()
# m.atualiza_modelo(days_for_update=0)
m.salva_modelo()
f = m.forecast()
x = m.modelo.get_params()
