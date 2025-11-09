import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load datasets ---
train_df = pd.read_csv('sales_dataset.csv')
test_df = pd.read_csv('sales_test_dataset.csv')

# --- Preprocessing ---
def prepare_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    return df[['Date', 'Sales', 'Price', 'Discount', 'Month']]

train = prepare_data(train_df)
test = prepare_data(test_df)

# --- Store results ---
results = {}

# ===========================
# 📈 LINEAR REGRESSION
# ===========================
lr = LinearRegression()
lr.fit(train[['Price', 'Discount', 'Month']], train['Sales'])
lr_preds = lr.predict(test[['Price', 'Discount', 'Month']])

results['Linear Regression'] = {
    'RMSE': np.sqrt(mean_squared_error(test['Sales'], lr_preds)),
    'R2': r2_score(test['Sales'], lr_preds)
}

# ===========================
# 🌲 RANDOM FOREST
# ===========================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train[['Price', 'Discount', 'Month']], train['Sales'])
rf_preds = rf.predict(test[['Price', 'Discount', 'Month']])

results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(test['Sales'], rf_preds)),
    'R2': r2_score(test['Sales'], rf_preds)
}

# ===========================
# 📉 ARIMA (on Sales only)
# ===========================
arima_train = train['Sales']
arima_model = ARIMA(arima_train, order=(5, 1, 0))
arima_result = arima_model.fit()

# Forecast next values (length of test data)
arima_preds = arima_result.forecast(steps=len(test))
results['ARIMA'] = {
    'RMSE': np.sqrt(mean_squared_error(test['Sales'], arima_preds)),
    'R2': r2_score(test['Sales'], arima_preds)
}

# ===========================
# 🔁 LSTM
# ===========================
# Scale features (excluding Date)
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train[['Sales', 'Price', 'Discount', 'Month']])
scaled_test = scaler.transform(test[['Sales', 'Price', 'Discount', 'Month']])

# Prepare LSTM input format (timesteps = 1)
X_train = np.reshape(scaled_train[:, 1:], (scaled_train.shape[0], 1, 3))
y_train = scaled_train[:, 0]
X_test = np.reshape(scaled_test[:, 1:], (scaled_test.shape[0], 1, 3))
y_test_actual = test['Sales'].values

# Build LSTM model
lstm = Sequential()
lstm.add(LSTM(64, activation='relu', input_shape=(1, 3)))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=50, verbose=0)

# Predict and inverse scale
lstm_preds_scaled = lstm.predict(X_test)
# Fill Sales column with LSTM preds, then inverse transform
temp = np.hstack((lstm_preds_scaled, scaled_test[:, 1:]))
lstm_preds = scaler.inverse_transform(temp)[:, 0]

results['LSTM'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_actual, lstm_preds)),
    'R2': r2_score(y_test_actual, lstm_preds)
}
def predict_sales(price, discount, month, model_choice):
    input_data = pd.DataFrame([[price, discount, month]], columns=['Price', 'Discount', 'Month'])

    if model_choice.lower() == 'linear':
        pred = lr.predict(input_data)[0]
    elif model_choice.lower() == 'random':
        pred = rf.predict(input_data)[0]
    elif model_choice.lower() == 'lstm':
        scaled_input = scaler.transform([[0, price, discount, month]])[:, 1:]
        reshaped_input = np.reshape(scaled_input, (1, 1, 3))
        lstm_scaled_pred = lstm.predict(reshaped_input)
        temp_input = np.hstack((lstm_scaled_pred, scaled_input))
        pred = scaler.inverse_transform(temp_input)[0][0]
    else:
        print("ARIMA or unsupported model. Use 'linear', 'random', or 'lstm'.")
        return None

    return pred
# ===========================
# 📊 Comparative Analysis (for Streamlit dashboard)
# ===========================
import io

def get_results_dataframe():
    """Return the comparative analysis results as a DataFrame."""
    df_results = pd.DataFrame(results).T
    return df_results

def get_comparison_plot():
    """Return the model comparison plot as a Matplotlib figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test['Date'], test['Sales'], label='Actual', color='black')
    ax.plot(test['Date'], lr_preds, label='Linear Regression')
    ax.plot(test['Date'], rf_preds, label='Random Forest')
    ax.plot(test['Date'], arima_preds, label='ARIMA')
    ax.plot(test['Date'], lstm_preds, label='LSTM')
    ax.legend()
    ax.set_title("Sales Prediction Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    return fig
