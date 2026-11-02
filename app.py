# Student Name: Fakrudeen Ali
# BITS ID: 2026AJ05480

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.title("Wine Quality Classification")
st.write("Student: Fakrudeen Ali | ID: 2026AJ05480")

uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# Directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check both root directory and 'model/' folder for compatibility
def get_path(filename):
    p_root = os.path.join(BASE_DIR, filename)
    p_sub = os.path.join(BASE_DIR, "model", filename)
    return p_sub if os.path.exists(p_sub) else p_root

model_paths = {
    "Logistic Regression": get_path("logistic_regression.pkl"),
    "Decision Tree": get_path("decision_tree.pkl"),
    "KNN": get_path("knn.pkl"),
    "Naive Bayes": get_path("naive_bayes.pkl"),
    "Support Vector Machine": get_path("support_vector_machine.pkl")
}
scaler_path = get_path("scaler.pkl")

selected_model = st.sidebar.selectbox("Choose Model", list(model_paths.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head(5))

    target_col = st.sidebar.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)

    if st.sidebar.button("Run Evaluation"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = joblib.load(model_paths[selected_model])
        scaler = joblib.load(scaler_path)

        if selected_model in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            X_eval = scaler.transform(X)
        else:
            X_eval = X

        preds = model.predict(X_eval)

        st.write(f"### Performance: {selected_model}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{accuracy_score(y, preds):.4f}")
        
        auc_val = "N/A"
        if hasattr(model, "predict_proba"):
            auc_val = f"{roc_auc_score(y, model.predict_proba(X_eval)[:, 1]):.4f}"
        c2.metric("AUC", auc_val)
        
        c3.metric("Precision", f"{precision_score(y, preds, average='weighted', zero_division=0):.4f}")
        c4.metric("Recall", f"{recall_score(y, preds, average='weighted', zero_division=0):.4f}")
        c5.metric("F1", f"{f1_score(y, preds, average='weighted', zero_division=0):.4f}")
        c6.metric("MCC", f"{matthews_corrcoef(y, preds):.4f}")

        st.write("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            st.pyplot(fig)

        with col_b:
            st.write("#### Classification Report")
            cr = classification_report(y, preds, output_dict=True)
            st.dataframe(pd.DataFrame(cr).transpose().round(4))
else:
    st.info("Upload test_data.csv to begin.")
