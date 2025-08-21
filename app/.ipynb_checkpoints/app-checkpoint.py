# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------
# Load model + preprocessing artifacts
# ------------------------
model = joblib.load("/Users/akashbhat/credit-card-fraud-detection/data/fraud_model.pkl")
scaler = joblib.load("/Users/akashbhat/credit-card-fraud-detection/data/scaler.pkl")
feature_columns = joblib.load("/Users/akashbhat/credit-card-fraud-detection/data/feature_columns.pkl")

# ------------------------
# Streamlit UI
# ------------------------
st.title("Credit Card Fraud Detection")

st.markdown("""
Upload a CSV file with transaction data.  
The file should include the following columns: Time, V1–V28, Amount, Class
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded CSV columns:", list(user_data.columns))

        # ------------------------
        # Step 1: Drop unused columns
        # ------------------------
        user_data = user_data.drop(columns=["Time", "Class"], errors="ignore")

        # ------------------------
        # Step 2: Scale Amount
        # ------------------------
        if "Amount" in user_data.columns:
            user_data["Amount_scaled"] = scaler.transform(user_data[["Amount"]])
            user_data = user_data.drop(columns=["Amount"], errors="ignore")

        # ------------------------
        # Step 3: Align columns to match training
        # ------------------------
        # Reorder columns and fill any missing ones with 0
        for col in feature_columns:
            if col not in user_data.columns:
                user_data[col] = 0  # or np.nan, depending on your model tolerance
        user_data = user_data[feature_columns]

        # ------------------------
        # Step 4: Make predictions
        # ------------------------
        proba = model.predict_proba(user_data)[:, 1]
        predictions = ["Fraud" if p > 0.5 else "Not Fraud" for p in proba]

        # Show results
        user_data["Prediction"] = predictions
        user_data["Fraud_Probability"] = proba
        st.write(user_data)

    except Exception as e:
        st.error(f"Error while processing file: {e}")
