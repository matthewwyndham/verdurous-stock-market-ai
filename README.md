# verdurous-stock-market-ai
An AI project in Python3 to predict the stock market

*You will need*:
- Pycharm
- alpha_vantage
- tensorflow
- graphviz
- urllib3
- pandas
- matplotlib
- numpy
- python3
- jupyter notebook

```python
import time
import csv  
from myapikey import APIkey
from alpha_vantage.timeseries import TimeSeries
import pickle
import numpy as np
import tensorflow as tf 

# Read CSV
def open_csv(path_to_csv):
    data = []
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] != '':
                data.append(row)
    return data

# Format dataset with classification (up/down)
def add_classification(stocks):
    new_stocks = []
    last_price = stocks[len(stocks) - 1][4]
    new_stocks.append(np.insert(stocks[len(stocks) - 1], 5, 1)) # up
    for s in reversed(list(range(len(stocks) - 1))):
        if (stocks[s][4] - last_price) > 0:
            new_stocks.append(np.insert(stocks[s], 5, 1)) # up
        else:
            new_stocks.append(np.insert(stocks[s], 5, 0)) # down
        last_price = stocks[s][4]
    return new_stocks

# Format dataset with classification (up/down)
def split_sample(data):
    sample = []
    for start in range(0, len(data) - 50, 50):
        sample.append(data[start:start+50])
    return sample

# Format dataset with classification (up/down)
def prepare(group_of_stocks):
    prepared_data_set = []
    for stock in group_of_stocks:
        prepared_data_set.append(split_sample(add_classification(stock[0].to_numpy())))
    return prepared_data_set

# Format dataset with classification (up/down)
def prep_part_two(group_of_stocks):
    new_set = []
    for stock in group_of_stocks:
        for collection in stock:
            classification = collection[0][5]
            training_data = collection[1:]
            new_set.append([training_data, classification])
    return new_set

# Access API
def get_daily(stock):
    ts = TimeSeries(key=APIkey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    return data, meta_data

def get_daily_compact(stock):
    ts = TimeSeries(key=APIkey, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=stock, outputsize='compact') # this returns 100 datapoints
    return data

# Wait for API (free) access limit
def batch_get_daily(stock_list):
    print("total (", len(stock_list), ")")
    big_counter = 0
    count = 0
    data = []
    for stock in stock_list:
        if count == 5:
            time.sleep(65) # 65 seconds just in case the timing on the server or here isn't perfect
            count = 0
        data.append(get_daily(stock))
        big_counter += 1
        print(stock, ":", big_counter, end=" ")
        count += 1
    return data

# example training
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(49, 6)),
    tf.keras.layers.Dense(512, activation=tf.nn.softmax),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(train_data), np.array(train_class), epochs=7)

# Test
model.evaluate(np.array(test_data), np.array(test_class))

# Real Life Prediction
chosen_stock = "AAPL"
rlp_stock = get_daily_compact(chosen_stock)
rlp_data = split_sample(add_classification(rlp_stock.to_numpy()))[0]
rlp_data = [rlp_data[:49]]

prediction = model.predict_classes(np.array(rlp_data), batch_size=1, verbose=0)

# Print results of prediction

# 1 = tomorrow will be up
# 0 = tomorrow will be down

if (prediction == 1):
    print(chosen_stock + " will go up tomorrow")
else:
    print(chosen_stock + " will go down tomorrow")

```