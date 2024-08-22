import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('new_scaler.pkl')  # Load the new scaler pipeline

# Function to preprocess and standardize input features
def preprocess_input(input_data, scaler):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=['gender', 'senior_citizen', 'tenure', 'monthly_charges', 'total_charges', 'contract'])
    # Transform the data using the scaler pipeline
    processed_data = scaler.transform(input_df)
    return processed_data

# Streamlit app interface
st.title("Churn Prediction")
st.write("Enter customer details to predict churn")

# Input fields
gender = st.selectbox("Gender", ("Male", "Female"))
senior_citizen = st.selectbox("Senior Citizen", (0, 1))
tenure = st.number_input("Tenure", min_value=0, max_value=72, value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)
contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))

# Preprocess inputs
if st.button("Predict Churn"):
    input_data = [gender, senior_citizen, tenure, monthly_charges, total_charges, contract]
    processed_data = preprocess_input(input_data, scaler)
    prediction = model.predict(processed_data)

    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
