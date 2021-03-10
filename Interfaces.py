# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:31:46 2021

@author: F8564619
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class AbstractModelo(ABC):
    
    def __inti__(self, ativo):
        self.__nome_modelo = ativo
    
    @property
    @abstractmethod
    def nome_modelo(self)->str:
        pass
    
    @abstractmethod
    def ajusta_dados(self):
        pass 
    
    @abstractmethod
    def fit(self)-> None:
        pass

    @abstractmethod
    def predict(self) -> np.array:
        pass

    @abstractmethod
    def salva_modelo(self):
        pass
    
    @abstractmethod
    def atualiza_modelo(self):
        pass
    
    
#%%
class ModeloExemplo(AbstractModelo):
    
    def __init__(self, df, ativo):
        self.modelo = 'modelo'
        self.ativo = ativo
        self.__nome_modelo = self.ativo
    
    @property
    def nome_modelo(self):
        return f'modelo_{self.__nome_modelo}'
    
    def ajusta_dados(self):
        pass 
    
    def fit(self)-> None:
        pass

    def salva_modelo(self):
        print('salvando')
        
    def predict(self):
        pass

    
    def atualiza_modelo(self):
        pass
        
#%%
m = ModeloExemplo('df', 'bbas')

print(m.nome_modelo)

