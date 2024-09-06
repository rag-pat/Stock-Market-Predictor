# Already Downloaded
import csv
import requests
import os
import math

# Need to be Downloaded
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


API_KEY = 'TLM2R8MSSTASS3Q0'
FILE_NAME = 'stock_data.csv'

def save_data_to_csv(symbol, API_KEY, FILE_NAME):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    print(symbol)

    if 'Time Series (Daily)' not in data:
        print("Error fetching data; Max requests reached for the API")
        return False
    else:
        data = data['Time Series (Daily)']
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            for timestamp, values in data.items():
                row = [timestamp, values['1. open'], values['2. high'], values['3. low'], values['4. close'], values['5. volume']]
                writer.writerow(row)

        print("Daily stock data has been saved to 'stock_data.csv'")
        return True


def get_train_test_data(FILE_NAME, dataset_indicator = "close"):
    df = pd.read_csv(FILE_NAME)
    dates = df.filter(['timestamp']).values
    dataset = df.filter([dataset_indicator]).values

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(dataset)

    SEQ_LEN = 60
    train_data_len = math.ceil(len(scaled_data) * .80)
    train_data = scaled_data[:train_data_len, :]
    x_train, y_train = [], []
    
    test_data = scaled_data[(train_data_len - SEQ_LEN) :, :]
    x_test, y_test = [], []

    for i in range(SEQ_LEN, len(train_data)):
        x_train.append(train_data[i - SEQ_LEN : i, :])
        y_train.append(train_data[i, :])

    for i in range(SEQ_LEN, len(test_data)):
        x_test.append(test_data[i - SEQ_LEN : i, :])
        y_test.append(test_data[i, :])
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    return x_train, y_train, x_test, y_test, scaler, dates

def create_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer="adam", loss="mse", metrics=["mse", 'msle'])
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_train, y_train))
    
    return model

def main(symbol):
    if not save_data_to_csv(symbol, API_KEY, FILE_NAME):
        return {"predictions": "error fetching data"}
    x_train, y_train, x_test, y_test, scaler, dates = get_train_test_data(FILE_NAME)
    model = create_model(x_train, y_train)
    
    predictions_2d = model.predict(x_test)
    predictions_2d = scaler.inverse_transform(predictions_2d)

    predictions = [price[0] for price in predictions_2d.tolist()]
    model.evaluate(x=x_test, y=y_test)
    
    
    df = pd.read_csv(FILE_NAME)
    actual = df.filter(["close"]).values
    
    dates = dates.tolist()
    actual = actual.tolist()
    
    for i in range(len(actual)):
        actual[i] = actual[i][0]
        dates[i] = dates[i][0]
        
    actual.reverse()
    dates.reverse()
    
    return {"predictions": predictions, "actual": actual, "dates": dates}
    

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8000/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    symbol: str

@app.post("/predict")
async def predict(item: Item):
    item_dict = item.dict()
    symbol = item_dict['symbol']
    temp = main(symbol)
    return temp