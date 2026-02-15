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
# Custom Styling for Tabs
# ==============================

st.markdown("""
<style>
button[data-baseweb="tab"] {
    font-size: 20px !important;
    font-weight: bold !important;
    padding: 12px 25px !important;
}
button[data-baseweb="tab"] {
    color: #444444 !important;
}
button[aria-selected="true"] {
    color: white !important;
    border-radius: 8px 8px 0px 0px !important;
}
button[data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    background-color: #e74c3c !important;
}
button[data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    background-color: #3498db !important;
}
button[data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
    background-color: #2ecc71 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Models
# ==============================

try:
    models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree Classifier": joblib.load("model/decision_tree_classifier.pkl"),
    "K-Nearest Neighbor Classifier": joblib.load("model/k-nearest_neighbor_classifier.pkl"),
    "Naive Bayes Classifier": joblib.load("model/naive_bayes_classifier.pkl"),
    "Random Forest (Ensemble)": joblib.load("model/random_forest_ensemble.pkl"),
    "XGBoost (Ensemble)": joblib.load("model/xgboost_ensemble.pkl"),
   }

    label_encoders = joblib.load("model/label_encoders.pkl")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==============================
# Sidebar (Only Inputs Here)
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
# Load Dataset
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
# Encode Data
# ==============================

for col in data.columns:
    if col in label_encoders:
        data[col] = label_encoders[col].transform(data[col])

X = data.drop("income", axis=1)
y = data["income"]

# ==============================
# Model Comparison (MUST COME BEFORE BEST MODEL)
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

# Add Ranking based on F1 Score
comparison_df["Rank"] = comparison_df["F1 Score"].rank(
    ascending=False,
    method="dense"
).astype(int)

# Sort by Rank
comparison_df = comparison_df.sort_values("Rank")


# ==============================
# NOW Show Best Model in Sidebar
# ==============================

best_row = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
best_model_name = best_row["Model"]
best_f1 = best_row["F1 Score"]

with st.sidebar:
    st.markdown("---")
    st.header("🏆 Best Model")
    st.markdown(f"""
    <div style="
        background-color:#d4edda;
        padding:15px;
        border-radius:10px;
        border-left:6px solid #28a745;
    ">
    <b>{best_model_name}</b><br>
    F1 Score: <b>{best_f1:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# Show Comparison Table
# ==============================

st.markdown("## 📊 Model Comparison")
st.dataframe(comparison_df, use_container_width=True)

# ==============================
# Detailed Evaluation
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
# Tabs
# ==============================

tab1, tab2, tab3 = st.tabs([
    "**Confusion Matrix**",
    "**Classification Report**",
    "**Predictions**"
])

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

    report_df.columns = [col.capitalize() for col in report_df.columns]
    report_df.index = [str(idx).capitalize() for idx in report_df.index]

    styled_report = (
        report_df.style
        .format("{:.4f}")
        .set_table_styles([
            {
                "selector": "th.col_heading",
                "props": [
                    ("background-color", "#2E86C1"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("font-size", "16px")
                ]
            },
            {
                "selector": "th.row_heading",
                "props": [
                    ("background-color", "#D6EAF8"),
                    ("color", "#1B4F72"),
                    ("font-weight", "bold"),
                    ("font-size", "15px")
                ]
            }
        ])
    )

    st.dataframe(styled_report, use_container_width=True)

with tab3:
    result_df = data.copy()
    result_df["Predicted Income"] = y_pred

    styled_predictions = (
        result_df.style
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#28a745"),
                    ("color", "white"),
                    ("font-size", "15px"),
                    ("font-weight", "bold")
                ]
            }
        ])
    )

    st.dataframe(styled_predictions, use_container_width=True)
