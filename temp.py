import csv
import requests

API_KEY = 'X9D8GWUD9AGGWXTR' # STORE ELSEWHERE
symbol = 'AAPL' # GET INPUT

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}' # MORE LINKS?
response = requests.get(url)
data = response.json()


if 'Time Series (Daily)' not in data:
    print("Error fetching data")
else:
    data = data['Time Series (Daily)']
    with open('stock_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        for timestamp, values in data.items():
            row = [timestamp, values['1. open'], values['2. high'], values['3. low'], values['4. close'], values['5. volume']]
            writer.writerow(row)

    print("Daily stock data has been saved to 'stock_data.csv'")
