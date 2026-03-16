import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Task: Time Series Analysis
df = pd.read_csv("StockPrices.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()
print("Columns:", df.columns)

# Use the correct date column
date_col = 'date'
value_col = 'close'  

# Convert 'date' column to datetime
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  

# Drop rows where date conversion failed
df = df.dropna(subset=[date_col])

# Set date as index
df.set_index(date_col, inplace=True)

# Sort by date just in case
df = df.sort_index()

# Plot the original time series
plt.figure(figsize=(12,5))
plt.plot(df[value_col], color='blue')
plt.title('Original Time Series - Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Decompose the time series
# Assuming monthly data or daily data, period can be adjusted
decomposition = seasonal_decompose(df[value_col], model='additive', period=30) 
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposition
plt.figure(figsize=(12,10))

plt.subplot(411)
plt.plot(df[value_col], label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Trend', color='orange')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Residuals', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Moving Average Smoothing
df['MA_7'] = df[value_col].rolling(window=7).mean()   
df['MA_30'] = df[value_col].rolling(window=30).mean() 

plt.figure(figsize=(12,5))
plt.plot(df[value_col], label='Original', color='blue')
plt.plot(df['MA_7'], label='7-Day MA', color='orange')
plt.plot(df['MA_30'], label='30-Day MA', color='green')
plt.title('Moving Average Smoothing - Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()