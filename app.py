import streamlit as st
import pandas as pd
import numpy as np
from prediction_model import predict_sales, get_results_dataframe, get_comparison_plot

st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")

st.title("📊 Sales Prediction Dashboard")

# Tabs for dashboard and prediction
tab1, tab2 = st.tabs(["📈 Model Comparison", "🔮 Predict Sales"])

with tab1:
    st.subheader("Comparative Model Performance")
    df_results = get_results_dataframe()
    st.dataframe(df_results)

    st.subheader("Prediction vs Actual Comparison")
    fig = get_comparison_plot()
    st.pyplot(fig)

with tab2:
    st.subheader("Sales Prediction")
    st.write("Enter the values below to predict future sales:")

    price = st.number_input("Enter Price:", min_value=0.0, value=100.0)
    discount = st.number_input("Enter Discount:", min_value=0.0, value=10.0)
    month = st.slider("Select Month (1-12):", 1, 12, 1)
    model_choice = st.selectbox("Choose Model:", ["Linear", "Random", "LSTM"])

    if st.button("Predict"):
        prediction = predict_sales(price, discount, month, model_choice)
        st.success(f"Predicted Sales using {model_choice} Model: {round(prediction, 2)}")
