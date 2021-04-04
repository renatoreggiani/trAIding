# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:31:46 2021

@author: F8564619
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
import mensageria
import os
from datetime import datetime


class AbstractModelo(ABC):

    def __inti__(self, ticker):
        self.__nome_modelo

    @property
    @abstractmethod
    def nome_modelo(self) -> str:
        pass

    @abstractmethod
    def ajusta_dados(self):
        pass

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def forecast(self) -> float:
        pass

    def carrega_modelo(self):
        try:
            self.modelo = pickle.load(open(self.__binfilename, 'rb'))
            mensageria.msg_loading_model(self.nome_modelo, self.ticker, self.__binfilename)
        except FileNotFoundError:
            mensageria.msg_file_not_found(self.__binfilename)

    @abstractmethod
    def salva_modelo(self):
        mensageria.msg_saving_model(self.nome_modelo, self.ticker, self.__binfilename)
        pickle.dump(self.modelo, open(self.__binfilename, "wb"))

    @abstractmethod
    def atualiza_modelo(self, days_for_update):
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


'''''
class ModeloExemplo(AbstractModelo):

    def __init__(self, df,  ticker):
        self.modelo = 'modelo'
        self.ticker = ticker
        self.__nome_modelo = self.ticker
        self.df = df

    @property
    def nome_modelo(self):
        return f'ARIMA_{self.__nome_modelo}'

    def ajusta_dados(self):
        pass

    def fit(self)-> None:
        pass

    def salva_modelo(self):
        #
        print('salvando')

    def forecast(self) -> float:
        pass

    def atualiza_modelo(self):
        pass

    def carrega_modelo(self):
        pass

'''
