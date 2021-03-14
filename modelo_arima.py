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
import os
import pickle
import json
import datetime

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
        #exclui todas as colunas exceto data e y
        self.df.dropna(subset=['y'], inplace=True)
        for col in self.df.columns: 
            if col not in ['date','y']:
                self.df.drop(col,axis='columns',inplace=True)

    def fit(self)->None:
        is_stat = stationary_test(self.df)
        arima_model = auto_arima(self.df, stationary=is_stat, start_p=0, start_d=0, start_q=0, max_p=10, max_d=10, max_q=10, 
                             start_P=0, start_D=0, start_Q=0, max_P=15, max_D=15, max_Q=15, m=15, seasonal='False', n_fits=50,
                             error_action='warn', trace=True, suppress_warnings=True, stepwise=False, random_state=20, n_jobs=-1)
        self.modelo = arima_model.fit(disp=0)

    def salva_modelo(self):
        try:
            with open('modelos/arima.json','r') as jfile:
                jdata = json.load(jfile)
        except:
            jdata = {}
        hoje = datetime.datetime.now().strftime("%Y-%m-%d")
        jdata[self.ticker].update({"parametros":self.modelo.order + self.modelo.seasonal_order,"train_date":hoje})
        json.dump(jdata, open('modelos/arima/arima.json','w'), indent=4)
        pickle.dump(self.modelo, open("modelos/arima/arima."+self.ticker+".bin","wb"))

    def forecast(self)->float:
        self.modelo.forecast()[0]

    def atualiza_modelo(self):
        train_date = datetime.datetime.strptime(os.path.getmtime("modelos/arima/arima."+self.ticker+".bin"))
        if ((datetime.datetime.now() - train_date).days) > self.days_force_update:
            self.fit()
    
    def carrega_modelo(self):
        self.modelo = pickle.load(open("modelos/arima/arima."+self.ticker+".bin",'rb'))