# Stock Market prediction with Machine Learning
# Senior Project at BYU-I by Matt Wyndham
# get your own API Key from Alpha Vantage: https://www.alphavantage.co/documentation/

from myapikey import APIkey
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

ts = TimeSeries(key=APIkey, output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')
data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.show()
