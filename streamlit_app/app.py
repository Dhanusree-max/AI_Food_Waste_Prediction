import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI-Powered Food Waste Prediction",
    page_icon="🍎",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("../models/food_waste_model.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("🍎 AI-Powered Food Waste Prediction System")

st.write(
    "Upload the preprocessed inventory dataset to predict food waste using the trained Random Forest model."
)

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload X_preprocessed.csv",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(data.head())

    if st.button("Predict Food Waste"):

        # Prediction
        predictions = model.predict(data)

        # Add prediction column
        result = data.copy()
        result["Predicted_Waste_Quantity"] = predictions

        st.success("Prediction Completed Successfully!")

        # ============================
        # Prediction Summary
        # ============================

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predicted Waste",
            f"{result['Predicted_Waste_Quantity'].sum():.2f}"
        )

        col2.metric(
            "Average Predicted Waste",
            f"{result['Predicted_Waste_Quantity'].mean():.2f}"
        )

        col3.metric(
            "Maximum Predicted Waste",
            f"{result['Predicted_Waste_Quantity'].max():.2f}"
        )

        # ============================
        # Prediction Table
        # ============================

        st.subheader("Prediction Results")

        st.dataframe(result)

        # ============================
        # Download Button
        # ============================

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Predictions",
            data=csv,
            file_name="food_waste_predictions.csv",
            mime="text/csv"
        )