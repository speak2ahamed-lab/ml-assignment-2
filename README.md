# BITS Pilani - WILP Division
## M.Tech (AIML) - Machine Learning (Assignment - 2)

**Student Name:** Fakrudeen Ali  
**BITS ID:** 2026AJ05480  
**Submission Date:** 08-Nov-2026  

---

### 1. Mandatory Submission Links
- **GitHub Repository Link:** https://github.com/speak2ahamed-lab/ml-assignment-2
- **Live Streamlit App Link:** https://ml-assignment-2-fakrudeenali.streamlit.app/

---

### 2. Problem Statement
Prediction of wine quality levels based on 12 physicochemical attributes using 5 different machine learning classification algorithms.

---

### 3. Dataset Description
- **Dataset Source:** UCI Machine Learning Repository (Wine Quality Dataset)
- **Total Instances:** 6,497 rows
- **Total Features:** 12 numeric features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type)
- **Target Variable:** `target` (Binary Classification: 0 = Quality rating < 6, 1 = Quality rating >= 6)

---

### 4. Models Used and Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7485 | 0.8194 | 0.7487 | 0.7485 | 0.7486 | 0.4721 |
| **Decision Tree** | 0.7392 | 0.7812 | 0.7390 | 0.7392 | 0.7391 | 0.4530 |
| **KNN** | 0.7631 | 0.8250 | 0.7628 | 0.7631 | 0.7629 | 0.5034 |
| **Naive Bayes** | 0.6908 | 0.7601 | 0.6975 | 0.6908 | 0.6924 | 0.3685 |
| **Support Vector Machine** | 0.7762 | 0.8491 | 0.7760 | 0.7762 | 0.7761 | 0.5312 |

---

### 5. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performs well as a baseline linear model. Feature scaling is necessary to balance precision and recall across classes. |
| **Decision Tree** | Captures basic classification rules but shows lower generalization on unseen test splits due to variance. |
| **KNN** | Delivers strong accuracy and MCC after feature normalization, though sensitive to local neighborhood density. |
| **Naive Bayes** | Lowest performance due to correlated features (e.g., sulfur and acidity) violating feature independence assumptions. |
| **Support Vector Machine** | Yields the highest overall scores across AUC and MCC by finding the optimal hyperplane margin in high-dimensional space. |
| **Overall, Winner for your dataset?** | Support Vector Machine (highest AUC of 0.8491 and MCC of 0.5312). |
