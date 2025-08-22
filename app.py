import streamlit as st
import pandas as pd
import pickle

# Load the pickle model
with open("fraud_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
threshold = saved["threshold"]
features = saved["features"]

st.set_page_config(page_title="Fraud Detection App", page_icon="🚨", layout="wide")

st.title("💳 Fraud Detection System")

uploaded_file = st.file_uploader("Upload CSV with transaction data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure correct columns
    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        X = df[features]
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)

        df["Fraud_Probability"] = y_pred_prob
        df["Fraud_Prediction"] = y_pred

        st.write("📊 Predictions:")
        st.dataframe(df)

        frauds = df[df["Fraud_Prediction"] == 1]
        st.write(f"🚨 Total Frauds Detected: {len(frauds)}")
        st.dataframe(frauds)
