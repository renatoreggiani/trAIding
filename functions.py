#!/usr/bin/env python
# coding: utf-8

# # Função get_finance_data
#     Função para capturar os dados dos ativos, acrescentar ".SA" no final do ticker para ativos 
#     negociados na Bovespa, exemplo "PETR4.SA".
#     Exemplo url base da API: https://query1.finance.yahoo.com/v7/finance/options/PETR4.SA?date=20201222
#     Exemplo url scrape da API: https://finance.yahoo.com/quote/PETR4.SA
#     
# ###    Parameters
#     period: default '1y', periodos validos: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#     inteval: default '1d', intervalos validos: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1d, 5d, 1wk, 1mo, 3mo.
# 

# In[1]:


import yfinance as yf
from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def get_finance_data(ticker, period='1y', interval='1d'):
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval=interval)
    return df


# # Função get_forecast
#     Função que entrega as próximas previsões
#     
# ###    Parameters
#     df: data frame com os dados historicos
#     ar, i, ma: parâmteros ARIMA
#     next: Quantidade de previsões
#     col_ref: Coluna de referência

# In[3]:


def get_forecast(df, col_ref='Low', next=1, p=5, d=1, q=0):
    y = df[col_ref].values
    model = ARIMA(y, order=(p,d,q)).fit()
    forecast = model.forecast(steps=next)[0]
    return forecast


# # Função stationary_test
#     Stationarity is an important concept in time-series and any time-series data should undergo a stationarity test before proceeding with a model.
# 
# ###    Parameters
#     df: data frame com os dados historicos
#     col_ref: Coluna de referência

# In[4]:


def stationary_test(df, col_ref='Low'):
    adf_test = ADFTest(alpha=0.05)
    return adf_test.should_diff(df[col_ref])[1]


# In[5]:


def get_auto_arima(ticker, period='1y', interval='1d', col_ref='Low'):
    df = get_finance_data(ticker, period, interval)
    is_stat = stationary_test(df)
    train = df[col_ref][:len(df)-10]
    arima_model = auto_arima(train, start_p=0, d=1, stationary=is_stat, start_q=0, max_p=5, mas_d=5, max_q=5, start_P=0, D=1, 
                             start_Q=1, max_P=5, max_D=5, max_Q=5, m=12, seasonal=True, error_action='warn', trace=True, 
                             suppress_warnings=True, stepwise=True, random_state=20, n_fits=50)
    return arima_model.order


# 

# # Exemplo de uso
# 
#     my_ticker='HGLG11.SA'
#     data = get_finance_data(my_ticker)
#     my_arima = get_auto_arima(my_ticker)
#     AR,I,MA = my_arima
#     get_forecast(data, next=10, p=AR,d=I,q=MA)

# In[ ]:




