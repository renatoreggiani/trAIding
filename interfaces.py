# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:31:46 2021

@author: F8564619
"""
from __future__ import annotations
from abc import ABC, abstractmethod


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

    @abstractmethod
    def carrega_modelo(self):
        pass

    @abstractmethod
    def salva_modelo(self):
        pass

    @abstractmethod
    def atualiza_modelo(self, days_for_update):
        pass


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
