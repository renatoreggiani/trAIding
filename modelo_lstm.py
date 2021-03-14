# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:06:21 2021

@author: F8564619
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

import os

from functions import get_finance_data
from interfaces import AbstractModelo

#%%

class modelo_lstm(AbstractModelo):
    
    def __init__(self, df,  ticker):
        self.modelo = self.cria_modelo()
        self.ticker = ticker
        self.__nome_modelo = self.ticker
        self.df = df
    
    @property
    def nome_modelo(self):
        return f'LSTM_{self.__nome_modelo}'
    
    
    def cria_modelo(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                    loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
                else:
                    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model
    
    
    def ajusta_dados(self):
        pass 
    
    def fit(self)-> None:
        pass

    def salva_modelo(self):
        pass
        
    def forecast(self) -> float:
        pass

    def atualiza_modelo(self):
        pass
    
    def carrega_modelo(self):
        pass