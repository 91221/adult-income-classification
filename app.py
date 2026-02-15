import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==============================
# Page Config
# ==============================

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.markdown("""
# 💼 Adult Income Prediction Dashboard  
Upload your test CSV & evaluate different ML models.
""")

# ==============================
# Load Models
# ==============================

try:
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl")
    }
    label_encoders = joblib.load("model/label_encoders.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==============================
# Sidebar - User Inputs
# ==============================

with st.sidebar:
    st.header("🔍 Select Model")
    model_choice = st.selectbox(
        "Choose a classification model:",
        list(models.keys())
    )

    st.markdown("---")
    st.header("📂 Upload Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file (must contain 'income' column)",
        type=["csv"]
    )

model = models[model_choice]

# ==============================
# Dataset Upload & Evaluation
# ==============================

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        st.write("### 📋 Uploaded Dataset Preview")
        st.dataframe(data.head())

        if "income" not in data.columns:
            st.error("Uploaded dataset must contain 'income' target column.")
            st.stop()

        # Encode categorical columns
        for col in data.columns:
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])

        X = data.drop("income", axis=1)
        y = data["income"]

        # Predictions
        y_pred = model.predict(X)

        # For AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = y_pred

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        # ==============================
        # Display Summary Metrics
        # ==============================

        st.markdown("## 📊 Evaluation Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("AUC Score", f"{auc:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC Score", f"{mcc:.4f}")

        # ==============================
        # Tabs for Detailed Views
        # ==============================

        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Raw Predictions"])

        with tab1:
            st.header("📌 Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual <=50K", "Actual >50K"],
                columns=["Predicted <=50K", "Predicted >50K"]
            )
            st.dataframe(cm_df)

        with tab2:
            st.header("📄 Classification Report")
            report_dict = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df)

        with tab3:
            st.header("🔎 Raw Predictions")
            result_df = data.copy()
            result_df["Predicted Income"] = y_pred
            st.dataframe(result_df)

    except Exception as e:
        st.error(f"Error processing dataset: {e}")

else:
    st.info("🛈 Please upload a test dataset CSV file using the sidebar uploader.")
