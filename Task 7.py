#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data1 = pd.read_excel("C:\\Users\\Admin\\Desktop\\Task 7 dataset\\1st.xlsx")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data1.iloc[:,:-1])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

data1["PC1"] = pca_result[:,0]
data1["PC2"] = pca_result[:,1]
print(data1.head())

plt.scatter(data1["PC1"], data1["PC2"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2d visualization of iris dataset")
plt.show




# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

data2 = pd.read_excel("C:\\Users\\Admin\\Desktop\\Task 7 dataset\\2nd.xlsx")

data2["date"] = pd.to_datetime(data2["date"], errors='coerce')
data2.dropna(subset=['date', 'close'], inplace=True)

data2 = data2.sort_values(by='date')
data2.set_index("date", inplace=True)

data2 = data2.asfreq('B', method='ffill')

plt.figure(figsize=(12, 5))
plt.plot(data2.index, data2["close"], color='blue', label="Actual Closing Price")
plt.title("Closing Price of Tesla")
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

data = data2['close']

model = ARIMA(data, order=(1, 3, 2), enforce_invertibility=False)
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)

forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

plt.figure(figsize=(12, 5))
plt.plot(data.index, data, label='Actual Price', color='blue')
plt.plot(forecast_index, forecast, label='Forecasted Price', color='red')
plt.title("Tesla Closing Price Forecast (ARIMA 1,3,2)")
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")
plt.legend()
plt.grid(True)
plt.show()

y_pred = model_fit.predict(start=len(data)-30, end=len(data)-1, dynamic=False)
y_true = data[-30:]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
mape = (abs(y_true - y_pred) / y_true).mean() * 100

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("Mean Absolute Percentage Error (MAPE):", round(mape, 2), "%")


# In[ ]:




