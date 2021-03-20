# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:28:15 2021

@author: Pedro
"""

# In[]

import mensageria
import modelo_arima
import yfinance as yf

# In[]


def back_test(modelo, ticker):
    # ticker = 'BBAS3.SA'
    mensageria.msg_loading_finance_data(ticker)
    tkr = yf.Ticker(ticker)
    df = tkr.history(period='1y', interval='1d')
    df = df.rename(columns={'Close': 'y'}, inplace=False)
    history, test = df[0:-52], df[-52:]
    # history = train.tolist()
    predictions = []

    for t in range(len(test)):
        m = modelo_arima.modelo_arima(history, ticker, days_for_update=7)
        m.fit()
        yhat = m.forecast()
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        int('predicted=%f, observed=%f' % (yhat, obs))


# In[]


'''
X = df.values
train, test = X[0:-52], X[-52:]
history = train.tolist()
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(arima_order[0],arima_order[1],arima_order[2]))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0][0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    int('predicted=%f, observed=%f' % (yhat, obs))
'''
