# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 18-11-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Load the temperature dataset
data = pd.read_csv("/content/MLTempDataset_datetime_numeric.csv")

# Step 2: Select the temperature column
X = pd.to_numeric(data["Hourly_Temp"], errors="coerce")

# Remove missing values
X = X.dropna().reset_index(drop=True)

# Step 3: Set required variables
N = 1000
plt.rcParams["figure.figsize"] = [12, 6]

# Display original data
plt.plot(X)
plt.title("Original Hourly Temperature Data")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()

# Step 4: Plot ACF and PACF for original data
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(X, lags=35, ax=plt.gca())
plt.title("Original Data ACF")

plt.subplot(2, 1, 2)
plot_pacf(X, lags=35, ax=plt.gca())
plt.title("Original Data PACF")

plt.tight_layout()
plt.show()

# Step 5: Fit ARMA(1,1) model
arma11_model = ARIMA(
    X,
    order=(1, 0, 1)
).fit()

print("ARMA(1,1) Model Summary")
print(arma11_model.summary())

# Extract AR and MA coefficients
phi1_arma11 = arma11_model.params["ar.L1"]
theta1_arma11 = arma11_model.params["ma.L1"]

# Define ARMA(1,1) coefficients
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

# Simulate ARMA(1,1)
ARMA_1 = ArmaProcess(
    ar1,
    ma1
).generate_sample(nsample=N)

# Plot simulated ARMA(1,1)
plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)

plt.title("Simulated ARMA(1,1) Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.xlim([0, 500])

plt.show()

# Step 6: Plot ACF and PACF for ARMA(1,1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=35, ax=plt.gca())
plt.title("ARMA(1,1) ACF")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=35, ax=plt.gca())
plt.title("ARMA(1,1) PACF")

plt.tight_layout()
plt.show()

# Step 7: Fit ARMA(2,2) model
arma22_model = ARIMA(
    X,
    order=(2, 0, 2)
).fit()

print("ARMA(2,2) Model Summary")
print(arma22_model.summary())

# Extract AR and MA coefficients
phi1_arma22 = arma22_model.params["ar.L1"]
phi2_arma22 = arma22_model.params["ar.L2"]

theta1_arma22 = arma22_model.params["ma.L1"]
theta2_arma22 = arma22_model.params["ma.L2"]

# Define ARMA(2,2) coefficients
ar2 = np.array([
    1,
    -phi1_arma22,
    -phi2_arma22
])

ma2 = np.array([
    1,
    theta1_arma22,
    theta2_arma22
])

# Simulate ARMA(2,2)
ARMA_2 = ArmaProcess(
    ar2,
    ma2
).generate_sample(nsample=N * 10)

# Plot simulated ARMA(2,2)
plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)

plt.title("Simulated ARMA(2,2) Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.xlim([0, 500])

plt.show()

# Step 8: Plot ACF and PACF for ARMA(2,2)
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plot_acf(ARMA_2, lags=40, ax=plt.gca())
plt.title("ARMA(2,2) ACF")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, lags=40, ax=plt.gca())
plt.title("ARMA(2,2) PACF")

plt.tight_layout()
plt.show()

```
### OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="1352" height="655" alt="image" src="https://github.com/user-attachments/assets/c751be76-975b-49cd-afef-e5abe32e7e22" />


Partial Autocorrelation

<img width="1497" height="496" alt="image" src="https://github.com/user-attachments/assets/5ecbfd0c-1159-4693-a805-1e85f672c711" />


Autocorrelation

<img width="1491" height="498" alt="image" src="https://github.com/user-attachments/assets/ccf3afd8-df06-46d8-b788-212f7ef42586" />


SIMULATED ARMA(2,2) PROCESS:

<img width="1337" height="675" alt="image" src="https://github.com/user-attachments/assets/b834079e-3196-4beb-9bae-81d1c3ae1c0a" />


Partial Autocorrelation

<img width="1535" height="502" alt="image" src="https://github.com/user-attachments/assets/249ace6f-7a85-44ae-9969-c5e588b6de60" />


Autocorrelation

<img width="1510" height="485" alt="image" src="https://github.com/user-attachments/assets/0ec7338f-eff4-4f26-bfe1-136e888252da" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
