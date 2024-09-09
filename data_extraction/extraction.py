import yfinance as yf
import pandas as pd
import numpy as np
import os
import pandas_ta as ta

stocks = ['GOOGL','NFLX','SPOT','AMZN','AAPL','META','TSLA','NVDA',
          '^IXIC','^GSPC','XOM','LOW','UNH','CL','NEE','ESS',
          'MSFT','AVGO','JNJ','V','COST','WMT','MCD',
          'IBM','VZ','NKE']

features = ['Close','Close_pct_change','Volume','SMA','RSI','EMA','Bollinger_Mid',
            'Bollinger_Upper','Bollinger_Lower','MACD','Stochastic_Oscillator','CCI',
            'Aroon_Up','Aroon_Down','VWAP','Open','High','Low']

period = '10y'
data_dir = 'stock_data'
os.makedirs(data_dir, exist_ok=True)

def add_indicators(df):
    df['SMA'] = ta.sma(df['Close'], length=14)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

    bb = ta.bbands(df['Close'], length=20)
    df['Bollinger_Mid'] = bb['BBM_20_2.0']
    df['Bollinger_Upper'] = bb['BBU_20_2.0']
    df['Bollinger_Lower'] = bb['BBL_20_2.0']

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Stochastic_Oscillator'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
    df['CCI'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (0.015 * df['Close'].rolling(window=20).std())

    aroon = ta.aroon(df['High'], df['Low'], length=14)
    df['Aroon_Up'] = aroon.iloc[:, 0]
    df['Aroon_Down'] = aroon.iloc[:, 1]

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    df['Close_pct_change'] = df['Close'].pct_change() * 100 

    df = df[features] 
    df.bfill(inplace=True)
    return df

data_set = []


for stock in stocks:
    ticker = yf.Ticker(stock)
    df = yf.download(stock, period=period)
    df = df.drop(columns=['Stock Splits', 'Dividends','Adj Close'], errors='ignore')
    
    df = add_indicators(df)

    df['Stock'] = stock

    file_path = os.path.join(data_dir, f'data.csv')
    data_set.append(df)
    print(f'Saved {stock} data with technical indicators to data set')

combined_df = pd.concat(data_set)
combined_df.reset_index(drop=True, inplace=True)
combined_df.to_csv(file_path,index=False)
print("All stock data processed and saved.")
