from numpy import *
import pandas as pd
from myapikey import APIkey
from alpha_vantage.timeseries import TimeSeries
from urllib import *
import csv

def get_daily(stock):
    ts = TimeSeries(key=APIkey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    return data, meta_data


def add_classification():
    pass


def open_csv(path_to_csv):
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] != '':
                print(row)
