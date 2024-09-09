import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_seq(data,input,output):
    X = []
    y = []
    for i in range(len(data) - input - output):
        X.append(data[i:i+input])
        y.append(data[i + input : i + input + output,0]) # 0 index to predict close
    return np.array(X), np.array(y)

def get_train_data(inputL, outputL):
    stocks = ['GOOGL', 'NFLX', 'SPOT', 'AMZN', 'AAPL', 'META', 'TSLA', 'NVDA',
              '^IXIC', '^GSPC', 'XOM', 'LOW', 'UNH', 'CL', 'NEE', 'ESS',
              'MSFT', 'AVGO', 'JNJ', 'V', 'COST', 'WMT', 'MCD',
              'IBM', 'VZ', 'NKE']
    
    features = ['Close', 'Close_pct_change', 'Volume', 'SMA', 'RSI', 'EMA', 'Bollinger_Mid',
                'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'Stochastic_Oscillator', 'CCI',
                'Aroon_Up', 'Aroon_Down', 'VWAP', 'Open', 'High', 'Low']
    
    file_path = os.path.join('stock_data', 'data.csv')
    data = pd.read_csv(file_path)

    data_scaled = {}
    scalers = {}

    for stock in stocks:
        stock_data = data[data['Stock'] == stock]
        stock_data = stock_data.drop(columns=['Stock'])

        data_scaled[stock] = []  
        scalers[stock] = {}

        for feature in features:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            feature_data = stock_data[feature].values.reshape(-1, 1)
            scaled_feature = scaler.fit_transform(feature_data)
            data_scaled[stock].append(scaled_feature)

            if feature == 'Close':
                scalers[stock]['Close'] = scaler

    X = []
    y = []
    for df_list in data_scaled.values():
        df = np.concatenate(df_list, axis=1)  # Combine features into a single array
        X_stock, y_stock = create_seq(df, inputL, outputL)
        X.append(X_stock)
        y.append(y_stock)

    X = np.vstack(X)
    y = np.vstack(y)

    return train_test_split(X, y, test_size=0.2, shuffle=True), scalers


