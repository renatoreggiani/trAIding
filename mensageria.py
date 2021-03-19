# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:50:10 2021

@author: Pedro
"""

from datetime import datetime


def print_now():
    print(datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))


def msg_loading_finance_data(ticker):
    print_now()
    print('Loading finance data for ' + ticker + '\n')


def msg_starting_fitting(modelo, ticker):
    print_now()
    print('Fitting ' + modelo + ' for ' + ticker + '\n')


def msg_fitting_complete(modelo, ticker):
    print_now()
    print('Model ' + modelo + ' for ' + ticker + ' just fitted\n')


def msg_saving_model(modelo, ticker, filename):
    print_now()
    print('Saving ' + modelo + ' for ' + ticker + ' at:')
    print('./' + filename + '\n')


def msg_loading_model(modelo, ticker, filename):
    print_now()
    print('Loading ' + modelo + ' for ' + ticker + ' from:')
    print('./' + filename + '\n')


def msg_file_not_found(filename):
    print_now()
    print('File not found:')
    print('./' + filename + '\n')


def msg_checking_model_update(modelo, ticker):
    print_now()
    print('Checking needing of update to ' + modelo + ' for ' + ticker + '\n')


def msg_model_up_to_date(modelo, ticker, last_update):
    print_now()
    print('Model ' + modelo + ' for ' + ticker + ' is up to date!')
    print('Last update at ' + datetime.strftime(last_update, "%d/%m/%Y") + '\n')


def msg_model_out_of_date(modelo, ticker, last_update):
    print_now()
    print('Model ' + modelo + ' for ' + ticker + ' is out of date!')
    print('Last update at ' + datetime.strftime(last_update, "%d/%m/%Y") + '\n')
