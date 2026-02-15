import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
Compare multiple ML models and evaluate performance.
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
# Sidebar
# ==============================

with st.sidebar:
    st.header("🔍 Model Selection")
    model_choice = st.selectbox(
        "Choose model for detailed view:",
        list(models.keys())
    )

    st.markdown("---")
    st.header("📂 Test Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file (optional)",
        type=["csv"]
    )

# ==============================
# Load Dataset (Default + Upload Option)
# ==============================

if uploaded_file is None:
    try:
        data = pd.read_csv("test_data.csv")
        st.info("Using default test_data.csv")
    except:
        st.error("Default test_data.csv not found. Please upload a dataset.")
        st.stop()
else:
    data = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully.")

st.write("### 📋 Dataset Preview")
st.dataframe(data.head())

if "income" not in data.columns:
    st.error("Dataset must contain 'income' column.")
    st.stop()

# ==============================
# Encode Categorical Features
# ==============================

for col in data.columns:
    if col in label_encoders:
        data[col] = label_encoders[col].transform(data[col])

X = data.drop("income", axis=1)
y = data["income"]

# ==============================
# Model Comparison Section
# ==============================

comparison_results = []

for name, model in models.items():

    y_pred_temp = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob_temp = model.predict_proba(X)[:, 1]
    else:
        y_prob_temp = y_pred_temp

    comparison_results.append([
        name,
        accuracy_score(y, y_pred_temp),
        roc_auc_score(y, y_prob_temp),
        precision_score(y, y_pred_temp),
        recall_score(y, y_pred_temp),
        f1_score(y, y_pred_temp),
        matthews_corrcoef(y, y_pred_temp)
    ])

comparison_df = pd.DataFrame(
    comparison_results,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

st.markdown("## 📊 Model Comparison")
st.dataframe(comparison_df)

# ==============================
# Best Model Card
# ==============================

best_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
best_model_name = best_row["Model"]
best_f1 = best_row["F1 Score"]

st.markdown("## 🏆 Best Model")

st.markdown(f"""
<div style="
    background-color:#d4edda;
    padding:20px;
    border-radius:10px;
    border-left:6px solid #28a745;
">
<h3 style="color:#155724;">{best_model_name}</h3>
<p style="font-size:18px;">F1 Score: <b>{best_f1:.4f}</b></p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Detailed Evaluation (Selected Model)
# ==============================

st.markdown("## 🔎 Detailed Evaluation")

selected_model = models[model_choice]

y_pred = selected_model.predict(X)

if hasattr(selected_model, "predict_proba"):
    y_prob = selected_model.predict_proba(X)[:, 1]
else:
    y_prob = y_pred

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)

# ==============================
# Summary Metrics
# ==============================

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
# Tabs Section
# ==============================

tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Predictions"])

with tab1:
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{model_choice} Confusion Matrix")
    st.pyplot(fig)

with tab2:
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

with tab3:
    result_df = data.copy()
    result_df["Predicted Income"] = y_pred
    st.dataframe(result_df)
