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

st.set_page_config(page_title="Adult Income Classification", layout="centered")

st.title("💼 Adult Income Classification System")
st.markdown("Machine Learning Models for Predicting Income Level (>50K or <=50K)")

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
# Model Selection
# ==============================

st.subheader("🔍 Select Model")
model_choice = st.selectbox(
    "Choose a classification model:",
    list(models.keys())
)

model = models[model_choice]

# ==============================
# Dataset Upload
# ==============================

st.subheader("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing test data",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Dataset")
        st.dataframe(data.head())

        if "income" not in data.columns:
            st.error("Uploaded dataset must contain 'income' column as target.")
            st.stop()

        # Encode categorical columns
        for col in data.columns:
            if col in label_encoders:
                data[col] = label_encoders[col].transform(data[col])

        X = data.drop("income", axis=1)
        y = data["income"]

        # ==============================
        # Prediction
        # ==============================

        y_pred = model.predict(X)

        # For AUC we need probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = y_pred

        # ==============================
        # Evaluation Metrics
        # ==============================

        st.subheader("📊 Evaluation Metrics")

        accuracy = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**AUC Score:** {auc:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**MCC Score:** {mcc:.4f}")

        # ==============================
        # Confusion Matrix
        # ==============================

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual <=50K", "Actual >50K"],
            columns=["Predicted <=50K", "Predicted >50K"]
        )

        st.dataframe(cm_df)

        # ==============================
        # Classification Report
        # ==============================

        st.subheader("📄 Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)

    except Exception as e:
        st.error(f"Error processing dataset: {e}")

else:
    st.info("Please upload a test dataset CSV file to evaluate the model.")
