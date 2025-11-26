import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained models
svm = joblib.load("model_svm.pkl")
isvm = joblib.load("model_isvm.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection using SVM vs Incremental SVM")
st.write("This app compares traditional SVM with incremental SVM on fraud prediction.")

# -----------------------------
# USER INPUT AREA
# -----------------------------

st.header("📌 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

amt = col1.number_input("Transaction Amount", 1, 20000, 300)
age = col2.number_input("User Age", 18, 90, 30)
hour = col3.number_input("Transaction Hour (0–23)", 0, 23, 10)
city_pop = col1.number_input("City Population", 100, 2_000_000, 50000)
merch_lat = col2.number_input("Merchant Latitude", -90.0, 90.0, 40.5)
merch_long = col3.number_input("Merchant Longitude", -180.0, 180.0, -73.9)

# Create dataframe for prediction
input_data = pd.DataFrame({
    'amt': [amt],
    'age': [age],
    'hour': [hour],
    'city_pop': [city_pop],
    'merch_lat': [merch_lat],
    'merch_long': [merch_long]
})

scaled_input = scaler.transform(input_data)

# -----------------------------
# PREDICT
# -----------------------------

if st.button("Predict Fraud"):
    svm_pred = svm.predict(scaled_input)[0]
    isvm_pred = isvm.predict(scaled_input)[0]

    st.subheader("🔍 Prediction Results")

    colA, colB = st.columns(2)

    colA.metric("SVM Prediction", 
                "Fraud" if svm_pred == 1 else "Legit")

    colB.metric("Incremental SVM Prediction", 
                "Fraud" if isvm_pred == 1 else "Legit")

# -----------------------------
# PLOT COMPARISON
# -----------------------------

st.header("📊 Batch Training Comparison")

try:
    svm_times = np.load("svm_times.npy")
    isvm_times = np.load("isvm_times.npy")

    svm_acc = np.load("svm_acc.npy")
    isvm_acc = np.load("isvm_acc.npy")

    # -------------------------
    #   Training Time Plot
    # -------------------------

    st.subheader("⏱ Training Time per Batch")

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(svm_times)+1), svm_times, marker='o', label="SVM (Full Retrain)")
    ax1.plot(range(1, len(isvm_times)+1), isvm_times, marker='o', label="iSVM (Incremental)")
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Training Time (seconds)")
    ax1.legend()
    st.pyplot(fig1)

    # -------------------------
    #   Accuracy Plot
    # -------------------------

    st.subheader("🎯 Accuracy after Each Batch")

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, len(svm_acc)+1), svm_acc, marker='o', label="SVM Accuracy")
    ax2.plot(range(1, len(isvm_acc)+1), isvm_acc, marker='o', label="iSVM Accuracy")
    ax2.set_xlabel("Batch Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    st.pyplot(fig2)

except Exception as e:
    st.warning("Batch comparison files not found. Please run the training notebook first.")
    st.error(str(e))
