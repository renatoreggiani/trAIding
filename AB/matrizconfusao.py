# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 19:22:05 2021

@author: F8564619
"""


from sklearn.metrics import confusion_matrix

import pandas as pd

df = pd.read_csv(r'C:\Users\f8564619\Documents\GitHub\trAIding\dataframes\BBAS3.SA.csv', sep=';', decimal=',')


df['mc_true'] = df.Close.diff()[1:]
df['mc_pred'] = (df.predict.shift(1)[1:].values - df.Close.shift(1)[1:])


y_true = df.Close.diff()[1:] > 0
y_pred = (df.predict.shift(1)[1:].values - df.Close.shift(1)[1:]) > 0 

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

pd.DataFrame(confusion_matrix(y_true, y_pred), index=['subiu', 'caiu'], columns=['subiu', 'caiu'])

acc = (tp + tn)/(fn + tn + fp + tp)

taxa_ac_positivo = tp / (tp+fp)




df.Close.pct_change()[1:].describe()

import seaborn as sns
sns.boxplot(df.Close.pct_change()[1:])
