# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('/content/powerconsumption.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
plt.plot(data.index, data['PowerConsumption_Zone1'])
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1')
plt.title('Power Consumption Zone 1 Time Series')
plt.show()
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
check_stationarity(data['PowerConsumption_Zone1'])
plot_acf(data['PowerConsumption_Zone1'])
plt.show()
plot_pacf(data['PowerConsumption_Zone1'])
plt.show()
train_size = int(len(data) * 0.8)
train, test = data['PowerConsumption_Zone1'][:train_size], data['PowerConsumption_Zone1'][train_size:]
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Power Consumption Zone 1')
plt.title('SARIMA Model Predictions for Power Consumption Zone 1')
plt.legend()
plt.show()
```

### OUTPUT:

![Screenshot 2025-05-17 123727](https://github.com/user-attachments/assets/43087cf6-7ddc-4c1e-a690-4d0935492854)

![Screenshot 2025-05-17 123745](https://github.com/user-attachments/assets/1f3d1690-7d50-4a15-81b2-601e3d7cc5e1)

![Screenshot 2025-05-17 123804](https://github.com/user-attachments/assets/1e24177a-6979-40d6-a437-dbab64d76788)

![Screenshot 2025-05-17 123820](https://github.com/user-attachments/assets/fbd4a3a0-7961-447b-a8a0-79c35219e113)

![Screenshot 2025-05-17 123836](https://github.com/user-attachments/assets/f39d15df-ac7a-434a-843f-868cabb628aa)

![Screenshot 2025-05-17 123854](https://github.com/user-attachments/assets/b01f7b6f-7146-4a11-925c-1011290f6565)

### RESULT:
Thus the program run successfully based on the SARIMA model.
