import streamlit as st
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
    df['Date'] = pd.to_datetime(df['Date'])
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

arima_preds = arima_result.forecast(steps=len(test))
results['ARIMA'] = {
    'RMSE': np.sqrt(mean_squared_error(test['Sales'], arima_preds)),
    'R2': r2_score(test['Sales'], arima_preds)
}

# ===========================
# 🔁 LSTM
# ===========================
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train[['Sales', 'Price', 'Discount', 'Month']])
scaled_test = scaler.transform(test[['Sales', 'Price', 'Discount', 'Month']])

X_train = np.reshape(scaled_train[:, 1:], (scaled_train.shape[0], 1, 3))
y_train = scaled_train[:, 0]
X_test = np.reshape(scaled_test[:, 1:], (scaled_test.shape[0], 1, 3))
y_test_actual = test['Sales'].values

lstm = Sequential()
lstm.add(LSTM(64, activation='relu', input_shape=(1, 3)))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train, y_train, epochs=50, verbose=0)

lstm_preds_scaled = lstm.predict(X_test)
temp = np.hstack((lstm_preds_scaled, scaled_test[:, 1:]))
lstm_preds = scaler.inverse_transform(temp)[:, 0]

results['LSTM'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test_actual, lstm_preds)),
    'R2': r2_score(y_test_actual, lstm_preds)
}

# --- Prediction Function ---
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
        st.error("Unsupported model. Choose 'linear', 'random', or 'lstm'.")
        return None
    return pred

# ===========================
# STREAMLIT APP
# ===========================

st.set_page_config(page_title="Sales Prediction App", layout="wide")
st.title("🛒 Sales Forecasting Dashboard")
st.markdown("This app compares **Linear Regression**, **Random Forest**, **ARIMA**, and **LSTM** models on sales data.")

# --- Comparative Analysis Table ---
st.header("🔍 Model Performance Comparison")
df_results = pd.DataFrame(results).T
st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

# --- Prediction Form ---
st.header("📈 Predict Sales")
with st.form(key='prediction_form'):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        price = st.number_input('Price', min_value=0.0, value=50.0)
    with col2:
        discount = st.number_input('Discount', min_value=0.0, max_value=100.0, value=10.0)
    with col3:
        month = st.selectbox('Month', list(range(1, 13)), index=0)
    with col4:
        model_choice = st.selectbox('Model', ['Linear', 'Random', 'LSTM'])

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        prediction = predict_sales(price, discount, month, model_choice)
        if prediction is not None:
            st.success(f"🎯 Predicted Sales using {model_choice} model: **{round(prediction, 2)}** units.")

# --- Line Plot of Predictions ---
st.header("📊 Model Predictions vs Actual Sales")

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(test['Date'], test['Sales'], label='Actual', color='black', linewidth=2)
ax.plot(test['Date'], lr_preds, label='Linear Regression', linestyle='--')
ax.plot(test['Date'], rf_preds, label='Random Forest', linestyle='-.')
ax.plot(test['Date'], arima_preds, label='ARIMA', linestyle=':')
ax.plot(test['Date'], lstm_preds, label='LSTM', linestyle='-')
ax.legend()
ax.set_title('Sales Predictions by Different Models', fontsize=18)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.grid(True)
st.pyplot(fig)
