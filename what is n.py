import csv
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Replace with your Alpha Vantage API key
api_key = 'X9D8GWUD9AGGWXTR'
# Define the stock symbol
symbol = 'AAPL'
# Define the endpoint
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

# Send the request to Alpha Vantage
response = requests.get(url)
# Parse the JSON response
data = response.json()

# Check if the response contains the expected data
if 'Time Series (Daily)' not in data:
    print("Error fetching data")
else:
    # Extract time series data
    time_series = data['Time Series (Daily)']

    # Create a DataFrame from the data
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Save the data to a CSV file
    df.to_csv('daily_stock_data.csv')

    # Prepare the data for the model
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['4. close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make a prediction for the next day
    last_day = df['Days'].max()
    next_day = np.array([[last_day + 1]])
    predicted_price = model.predict(next_day)

    print(f"Predicted stock price for the next day: {predicted_price[0]}")

    # Optionally, plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Actual Prices')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
    plt.scatter(next_day, predicted_price, color='green', marker='x', s=100, label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
