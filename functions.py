#!/usr/bin/env python
# coding: utf-8

# In[8]:


import yfinance as yf
from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima



def get_finance_data(ticker, period='1y', interval='1d'):
    '''
    Função para capturar os dados dos ativos, acrescentar ".SA" no final do ticker para ativos 
    negociados na Bovespa, exemplo "PETR4.SA".
    Exemplo url base da API: https://query1.finance.yahoo.com/v7/finance/options/PETR4.SA?date=20201222
    Exemplo url scrape da API: https://finance.yahoo.com/quote/PETR4.SA
    
    Parameters
    ----------
    period: default '1y', periodos validos: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    inteval: default '1d', intervalos validos: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1d, 5d, 1wk, 1mo, 3mo.
    '''
    
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval)
    return df


def test_unit_root(s):
    adf = adfuller(s)
    return adf[0] > adf[4]['1%']

def stationary_test(s):
    adf_test = ADFTest(alpha=0.01)
    return not adf_test.should_diff(s)[1]
                           
def arima_forecast(s, next=1, p=5, d=1, q=0):
    '''
    Função que entrega as próximas previsões utilizando ARIMA
    
    Parameters
    ----------
    df: data frame com os dados historicos
    ar, i, ma: parâmteros ARIMA
    next: Quantidade de previsões
    col_ref: Coluna de referência
    '''    
    
    if test_unit_root(s):
        model = ARIMA(s, order=(p,d,q)).fit()
        forecast = model.forecast(steps=next)[0]
        return forecast
    else:
        print('Serie estacionaria, nao utilizar modelo ARIMA')



# # Função get_auto_arima
#     Retorna os valores ideais para calibração do ARIMA
# 
# ###    Parameters
#     ticker: ticker do papel
#     period: default '1y', periodos validos: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#     inteval: default '1d', intervalos validos: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1d, 5d, 1wk, 1mo, 3mo.
#     col_ref: default 'Low', Coluna de referência 

# In[5]:
'''
def get_auto_arima(s):
    is_stat = stationary_test(s)
    if test_unit_root(s):
        arima_model = auto_arima(s, stationary=is_stat, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, 
                                 start_Q=1, max_P=5, max_D=5, max_Q=5, m=12, seasonal=False, error_action='warn', trace=True, 
                                 suppress_warnings=True, stepwise=False, random_state=20, n_fits=50, n_jobs=-1)
        return arima_model
    else:
        print('Serie estacionaria, nao utilizar modelo ARIMA')
        
        
def get_auto_sarima(s):
    is_stat = stationary_test(s)
    if test_unit_root(s):
        arima_model = auto_arima(s, stationary=is_stat, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, 
                                 start_Q=1, max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, error_action='warn', trace=True, 
                                 suppress_warnings=True, stepwise=False, random_state=20, n_fits=50, n_jobs=-1)
        return arima_model
    else:
        print('Serie estacionaria, nao utilizar modelo SARIMA')
'''
def get_arima_model(s):
    is_stat = stationary_test(s)
    is_seasonal = test_unit_root(s) 
    arima_model = auto_arima(s, stationary=is_stat, start_p=0, d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, 
                             start_Q=1, max_P=5, max_D=5, max_Q=5, m=12, seasonal=is_seasonal, error_action='warn', trace=True, 
                             suppress_warnings=True, stepwise=False, random_state=20, n_fits=50, n_jobs=-1)
    return arima_model
