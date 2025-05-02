import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mydata.csv')

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

train_end = pd.to_datetime('2016-12-31')
df['Data Split'] = df['Date'].apply(lambda x: 'Train' if x <= train_end else 'Test')

" Temporal Distribution (Train Vs Test) Plot "
plt.figure(figsize=(14, 6))
for label, color in zip(['Train', 'Test'], ['tab:blue', 'tab:orange']):
    subset = df[df['Data Split'] == label]
    plt.hist(subset['Date'], bins=50, alpha=0.7, label=label, color=color)
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.title('Temporal Distribution of Training vs. Test Data')
plt.legend()
plt.tight_layout()
plt.show()

" Amazon Closing Price Plot "
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Close'])
plt.title('Amazon Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

" Daily Return Distribution Plot "
df['Return'] = df['Close'].pct_change()
plt.figure(figsize=(8,5))
plt.hist(df['Return'].dropna(), bins=50, color='skyblue')
plt.title('Distribution of Daily Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()

" SMA and EMA Closing Price Plot "
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['SMA'], label='SMA')
plt.plot(df['Date'], df['EMA'], label='EMA')
plt.title('Closing Price with SMA and EMA')
plt.legend()
plt.show()

"RSI Plot"
plt.figure(figsize=(14,3))
plt.plot(df['Date'], df['RSI'])
plt.title('Relative Strength Index (RSI)')
plt.show()

"Rolling Mean & Std Plot"
window = 30
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['Close'].rolling(window).mean(), label='Rolling Mean')
plt.plot(df['Date'], df['Close'].rolling(window).std(), label='Rolling Std')
plt.title('Rolling Mean & Std of Closing Price')
plt.legend()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['Close'].dropna(), lags=40)
plt.show()
plot_pacf(df['Close'].dropna(), lags=40)
plt.show()