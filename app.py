import streamlit as st
import pandas as pd
import numpy as np
from prediction_model import predict_sales

st.set_page_config(page_title="Sales Prediction Dashboard", layout="centered")

st.title("📈 Sales Prediction Web App")

st.write("Enter the values below to predict sales using different models:")

price = st.number_input("Enter Price:", min_value=0.0, value=100.0)
discount = st.number_input("Enter Discount:", min_value=0.0, value=10.0)
month = st.slider("Select Month (1-12):", 1, 12, 1)
model_choice = st.selectbox("Choose Model:", ["Linear", "Random", "LSTM"])

if st.button("Predict"):
    prediction = predict_sales(price, discount, month, model_choice)
    st.success(f"Predicted Sales using {model_choice} Model: {round(prediction, 2)}")
