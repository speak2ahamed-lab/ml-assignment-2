# BITS Pilani - WILP Division
## M.Tech (AIML) - Machine Learning (Assignment - 2)

**Student Name:** Fakrudeen Ali  
**BITS ID:** 2026AJ05480  
**Submission Date:** 08-Nov-2026  

---

### Mandatory Links
- **GitHub Repository Link:** https://github.com/speak2ahamed-lab/ml-assignment-2
- **Live Streamlit App Link:** https://ml-assignment-2-fakrudeenali.streamlit.app/

---

### A. Problem Statement
Prediction of wine quality levels based on 12 physicochemical attributes using 5 different machine learning classification algorithms.

---

### B. Dataset Description
- **Dataset Source:** UCI Machine Learning Repository (Wine Quality Dataset)
- **Total Instances:** 6,497 rows
- **Total Features:** 12 numeric features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type)
- **Target Variable:** `target` (Binary Classification: 0 = Quality rating < 6, 1 = Quality rating >= 6)

---

### C. Github Repository & Streamlit Link
- **GitHub Repository Link:** https://github.com/speak2ahamed-lab/ml-assignment-2
- **Live Streamlit App Link:** https://ml-assignment-2-fakrudeenali.streamlit.app/

---

### D. Models Used and Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7392 | 0.8057 | 0.7333 | 0.7392 | 0.7329 | 0.4214 |
| **Decision Tree** | 0.7500 | 0.7985 | 0.7499 | 0.7500 | 0.7499 | 0.4616 |
| **KNN** | 0.7408 | 0.8004 | 0.7363 | 0.7408 | 0.7373 | 0.4308 |
| **Naive Bayes** | 0.6831 | 0.7445 | 0.6714 | 0.6831 | 0.6710 | 0.2861 |
| **Support Vector Machine** | 0.7731 | 0.8400 | 0.7694 | 0.7731 | 0.7669 | 0.4971 |

---

### Model Observations on the performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Stable linear baseline (Accuracy: 0.7392, AUC: 0.8057). Requires feature scaling to prevent high-variance attributes from skewing class boundaries. |
| **Decision Tree** | Captures non-linear thresholds well with an accuracy of 0.7500 and balanced F1 (0.7499), but exhibits slightly lower AUC than margin-based classifiers. |
| **KNN** | Delivers competitive accuracy (0.7408) and AUC (0.8004) after standard feature scaling, though sensitive to local neighborhood density. |
| **Naive Bayes** | Lowest performer across all 6 metrics (Accuracy: 0.6831, AUC: 0.7445, MCC: 0.2861) because chemical features exhibit multicollinearity, violating feature independence assumptions. |
| **Support Vector Machine** | Best overall performer across all metrics (Accuracy: 0.7731, AUC: 0.8400, MCC: 0.4971) by establishing an optimal maximum-margin hyperplane in high-dimensional feature space. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** (highest AUC of 0.8400 and highest MCC of 0.4971). |
