#!/usr/bin/env python
# coding: utf-8

# In[1]:
    
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import json
import datetime

from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

plt.style.use('fivethirtyeight')

# In[2]:
    
def get_finance_data(ticker, period='max', interval='1d'):
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval)
    return df

def test_unit_root(s):
    adf = adfuller(s)
    return adf[0] > adf[4]['1%']

def stationary_test(s):
    adf_test = ADFTest(alpha=0.01)
    return not adf_test.should_diff(s)[1]

def get_arima_model(s, is_seasonal=False):
    is_stat = test_unit_root(s)
    arima_model = auto_arima(s, stationary=is_stat, start_p=0, start_d=0, start_q=0, max_p=10, max_d=10, max_q=10, start_P=0, start_D=0, 
                             start_Q=0, max_P=15, max_D=15, max_Q=15, m=15, seasonal=is_seasonal, error_action='warn', trace=True, 
                             suppress_warnings=True, stepwise=False, random_state=20, n_fits=50, n_jobs=-1)
    return arima_model

def get_arima_data(yticker):
    data_return = ''
    if os.path.isfile("files/db.json"):
        with open('files/db.json','r+') as jfile:
            jdata = json.load(jfile)
            if (yticker in jdata.keys()):
                if ("ARIMA" in jdata[yticker]):
                    data_return = jdata[yticker]["ARIMA"]["parametros"]
                else:
                    print("Data for "+yticker+" not found")
    return data_return

# In[20]:
    
def run_arima_coach(yticker_list, days_force_update=0):

    n_steps = 2
    must_run = False
    hoje = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if not os.path.isdir("files"):
        os.mkdir("files")
    
    for yticker in yticker_list:
        ticker = yticker
        if os.path.isfile("files/db.json"):
            with open('files/db.json','r+') as jfile:
                jdata = json.load(jfile)
                if (ticker in jdata.keys()):
                    print("Dados de "+ticker+" já existentes.")
                    if ("ARIMA" in jdata[ticker]):
                        print("Dados atualizados em "+jdata[ticker]["ARIMA"]["train_date"])
                        if ("train_date" in jdata[ticker]["ARIMA"]):
                            if (days_force_update > 0):
                                train_date = datetime.datetime.strptime(jdata[ticker]["ARIMA"]["train_date"], '%Y-%m-%d')
                                must_run = ((datetime.datetime.now() - train_date).days) > days_force_update
                        else:
                            must_run = True
                    else:
                        must_run = True
                else:
                    print("ticker "+ticker+" não encontrado")
                    must_run = True
                if must_run:
                    print("capturando dados do yFinance: "+yticker)
                    data = get_finance_data(yticker)
                    data.dropna(subset=['Low'], inplace=True)
                    train = data['Low'][:len(data)-n_steps+1]
                    test = data['Low'][-n_steps:]
                    jdata[ticker]={}
                    jdata[ticker].update({"yticker":yticker})
                    jfile.seek(0)
                    json.dump(jdata, jfile)
                    
        my_arima = []
        if "ARIMA" not in jdata[ticker]:
            print("rodando auto arima para "+ticker)
            arima_model = get_arima_model(train)
            my_arima.append(arima_model.order[0])
            my_arima.append(arima_model.order[1])
            my_arima.append(arima_model.order[2])
            my_arima.append(arima_model.seasonal_order[0])
            my_arima.append(arima_model.seasonal_order[1])
            my_arima.append(arima_model.seasonal_order[2])
            my_arima.append(arima_model.seasonal_order[3])
            with open('files/db.json','r+') as jfile:
                jdata[ticker].update({"ARIMA":{"parametros":my_arima,"train_date":hoje}})
                jfile.seek(0)
                json.dump(jdata, jfile)
        
        jfile.close()
        
# In[21]:
    
def do_arima_forecast(yticker):
    yticker = "BBFI11B.SA" #apagar
    #date_ini = datetime.datetime.now().strftime("%Y-%m-%d")
    o = get_arima_data(yticker)
    df_log = get_finance_data(yticker)
    df_log = df_log['Low']
    model = ARIMA(df_log, order=(o[0],o[1],o[2]))
    results = model.fit(disp=-1)
    return results.fittedvalues

# In[ ]:

teste = ["RNDP11.SA","OIBR3.SA","VILG11.SA","BBFI11B.SA","OIBR3.SA","PETR4.SA"]
run_arima_coach(teste, days_force_update=2)

# In[ ]:

fc = do_arima_forecast("RBBV11.SA")
df_log = get_finance_data("RBBV11.SA")
df_log = df_log['Low']
plt.plot(df_log[-5:], color='blue')
plt.plot(fc[-5:], color='red')

# In[8]:




# In[11]:





# In[13]:




# In[ ]:




