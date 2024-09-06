This project is a comprehensive web application that predicts stock prices using machine learning
and visualizes them through an interactive UI. It leverages a FastAPI backend to fetch and process
stock data from Alpha Vantage, and a frontend built with HTML, CSS, and JavaScript to visualize
the results. The model is trained using LSTM (Long Short-Term Memory) and Linear Regression to
predict future stock prices based on historical data.

Key Features:
- Data Fetching: Retrieves daily stock data using Alpha Vantage API and saves it to a CSV file.
- Machine Learning Models: Implements LSTM for deep learning-based predictions and Linear Regression for simpler predictions.
- Backend: FastAPI framework processes requests and generates predictions.
- Frontend: A responsive web interface using Bootstrap and Chart.js for interactive stock data visualization.
- Deployment: CORS middleware setup for easy integration with various frontend applications.

How it Works:
- Fetch Stock Data: Retrieve stock price data from Alpha Vantage API.
- Train Models: Prepare the data and train a machine learning model (LSTM or Linear Regression).
- Make Predictions: Use the trained model to predict future stock prices.
- Visualize Data: Display the stock data and predictions on a web interface using Chart.js.
