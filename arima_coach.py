#!/usr/bin/env python
# coding: utf-8

# In[1]:
    
import yfinance as yf
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt 
import os
import json
import datetime

from sklearn.metrics import mean_squared_error
from math import sqrt

from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')

# In[2]:
    
def get_finance_data(ticker, period='5y', interval='1d'):
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval)
    return df.dropna(subset=['Close'])

def test_unit_root(s):
    adf = adfuller(s)
    return adf[0] > adf[4]['1%']

def stationary_test(s):
    adf_test = ADFTest(alpha=0.01)
    return not adf_test.should_diff(s)[1]

def get_arima_model(s, is_seasonal=False):
    is_stat = not test_unit_root(s)
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

    #n_steps = 2
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
                        must_run = True
                else:
                    print("ticker "+ticker+" não encontrado")
                    must_run = True
                if must_run:
                    print("capturando dados do yFinance: "+yticker)
                    data = get_finance_data(yticker)
                    data.dropna(subset=['Close'], inplace=True)
                    #train = data['Close'][:len(data)-n_steps+1]
                    train = data['Close']
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
    
    #yticker = "BTC-USD" #apagar
    arima_order = get_arima_data(yticker)
    if arima_order:
        df_log = get_finance_data(yticker)
        df_log = df_log.dropna()['Close']
        model = ARIMA(df_log, order=(arima_order[0],arima_order[1],arima_order[2]))
        try:    
            model_fit = model.fit(disp=0)
        except Exception as e: 
            print(e)
            print("\n")
            return False 
        return model_fit.forecast()[0]
    else:
        return False

# In[21]:

ticker = "VILG11.SA"
def test_arima(tickers): 
    for ticker in tickers:
        #load dataframe
        series = get_finance_data(ticker)
        series = series.dropna()['Close']
        series = series[-100:]

        # split into train and test sets
        X = series.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()

        #get arima params
        arima_order = get_arima_data(ticker)

        # walk-forward validation
        for t in range(len(test)):
        	model = ARIMA(history, order=(arima_order[0],arima_order[1],arima_order[2]))
        	model_fit = model.fit(disp=0,transparams=False)
        	output = model_fit.forecast()
        	yhat = output[0]
        	predictions.append(yhat)
        	obs = test[t]
        	history.append(obs)

        # evaluate forecasts
        rmse = sqrt(mean_squared_error(test, predictions))
        print('Teste RMSE para '+ticker+': %.3f' % rmse)
        with open('dataframes/^^resumo.json','r+') as jfile:
            jdata = json.load(jfile)
            jdata[ticker]={}
            jdata[ticker].update({"RMSE":str(rmse)})
            jfile.seek(0)
            json.dump(jdata, jfile)  


# In[]:
    
def get_next_value(yticker):

    #yticker = "PETR4.SA" #apagar
    arima_order = get_arima_data(yticker)
    if arima_order:
        df_log = get_finance_data(yticker)
        df_log =  df_log.dropna()['Close']
        model = ARIMA(df_log, order=(arima_order[0],arima_order[1],arima_order[2]))
        model_fit = model.fit()
        return model_fit.forecast()[0]
    else:
        return False

# In[]:

ticker = "PETR4.SA"
def run_statistics(tickers, ganho_min=0.005, gap=0.995):
    for ticker in tickers:
         print(ticker)
         predict = do_arima_forecast(ticker)
         if isinstance(predict, bool):
             continue
         df_log = get_finance_data(ticker)
         df_log = df_log.drop(columns=['Dividends','Stock Splits','Volume'])
         df_log = df_log[1:-1]
         df_log['predict']=predict
         df_log['predict_pct'] = (df_log['predict']/df_log['Close'])-1
         
         ganho_min = df_log.Close.pct_change()[1:].describe()['50%']
         gap = 1-ganho_min
    
         #captura entradas
         entrada = pd.DataFrame()
         entrada['predict'] = df_log['predict_pct']
         entrada['predict'] = df_log[df_log['predict_pct']>ganho_min]['predict']*gap
         entrada['predict'] = entrada['predict'].shift(1)
         entrada['open'] = df_log[entrada['predict'].notnull()]['Open']
         df_log['entrada'] = entrada['predict'].combine(entrada['open'],min)
    
         #captura saida
         df_log['saida'] = df_log[df_log['entrada'].notnull()]['Close']
    
         #calculando lucro
         df_log['profit'] = (df_log['saida']/df_log['entrada'])-1
         df_log['profit'] = df_log['profit'].fillna(0)
         profit_day = df_log['profit'].mean()
         profit_month = ((1+profit_day) ** 20) -1
         
         #calculando assertividade da subida
         df_log['sucesso'] = (df_log['profit']>0) | (df_log['entrada'].isnull())
         df_log['subida'] = df_log[df_log['entrada'].notnull()]['profit']>0
         df_log.to_csv("dataframes/"+ticker+".csv",sep=";",decimal=",")
    
         resumo = ((df_log['sucesso'].value_counts(True))*100).round(2)
         assertivo = ((df_log['subida'].value_counts(True))*100).round(2)
         
         with open('dataframes/^^resumo.json','r+') as jfile:
             jdata = json.load(jfile)
             jdata[ticker]={}
             jdata[ticker].update({"Ganho pra acionamento":str(ganho_min),
                                   "Gap de desconto":str(gap),
                                   "Acertos Decisao":str(resumo[True]),
                                   "Acertos Subida":str(assertivo[True]),
                                   "Lucro medio diario":str(round(profit_day*100,2)),
                                   "Lucro medio mensal":str(round(profit_month*100,2))})
             jfile.seek(0)
             json.dump(jdata, jfile)
    
         print("Acertos Decisão: "+str(resumo[True])+"%")
         print("Acertos Subida: "+str(assertivo[True])+"%")
         print("Lucro médio diário:"+str(round(profit_day*100,2))+"%")
         print("Lucro médio mensal:"+str(round(profit_month*100,2))+"%\n")

# In[ ]:

#JPYEUR=X com provavel erro nos dados do YFinance
tickers = ["RNDP11.SA","OIBR3.SA","VILG11.SA","BBFI11B.SA", "PETR4.SA", "EUR=X", "BTC-USD", 
         "VALE3.SA", "BBAS3.SA", "ITUB3.SA","AAPL","GOOG","TSLA","^DJI","^GSPC","GC=F","CL=F","BZ=F"]
run_arima_coach(tickers, days_force_update=1)
#run_statistics(tickers)
test_arima(tickers)

# In[]:
    
