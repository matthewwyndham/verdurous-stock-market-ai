from numpy import *
import pandas as pd
from myapikey import APIkey
from alpha_vantage.timeseries import TimeSeries
from urllib import *


def get_daily(stock):
    ts = TimeSeries(key=APIkey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    return data, meta_data
