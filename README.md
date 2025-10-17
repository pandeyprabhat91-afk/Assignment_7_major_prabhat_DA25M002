# DA5401 Assignment 7: Multi-Class Model Selection using ROC and Precision-Recall Curves

**Author:** Major Prabhat Pandey (DA25M002)  
**Program:** M.Tech in Artificial Intelligence and Data Science  
**Date:** October 2025  

---

## 1. Project Overview

This project focuses on **multi-class model selection and evaluation** using **ROC (Receiver Operating Characteristic)** and **Precision-Recall curves** on the **UCI Landsat Satellite dataset**.  
The primary objective is to compare multiple classification models and determine the best-performing model based on robust evaluation metrics.

---

## 2. Objectives

1. Apply ROC and Precision-Recall Curves for multi-class classification using the One-vs-Rest strategy.  
2. Perform hyperparameter tuning using GridSearchCV (3-fold cross-validation).  
3. Evaluate models based on Accuracy, Weighted F1-Score, ROC-AUC, and Average Precision.  
4. Conduct hyperparameter sensitivity analysis and per-class ROC-AUC evaluation.  
5. Recommend the best-performing model supported by interpretability and analysis.

---

## 3. Dataset

- **Source:** UCI Statlog (Landsat Satellite) Dataset  
- **Training samples:** 4435  
- **Test samples:** 2000  
- **Features:** 36  
- **Classes:** 6 (Class 7 excluded as per assignment requirement)  

After filtering out Class 7, the dataset contains Classes **1–5**, representing distinct land-cover categories.

---

## 4. Models Implemented

| Model | Description |
|--------|-------------|
| KNN | k-Nearest Neighbors classifier |
| Decision Tree | Entropy-based tree classifier |
| Logistic Regression | Multinomial logistic regression with L2 regularization |
| SVM | Support Vector Machine with RBF kernel (`probability=True`) |
| Random Forest | Ensemble of decision trees |
| Naive Bayes | Gaussian Naive Bayes classifier |
| XGBoost | Gradient Boosting Classifier |
| Dummy (Prior) | Baseline model predicting the most frequent class |

---

## 5. Development Timeline

| Date | Activity |
|------|-----------|
| 17 Oct 2025 | Data loading, cleaning, and preprocessing |
| 18 Oct 2025 | GridSearchCV tuning for all models |
| 19 Oct 2025 | ROC computation using One-vs-Rest approach |
| 20 Oct 2025 | Precision-Recall curve computation and macro-averaging fixes |
| 21 Oct 2025 | Final visualizations and confusion matrix analysis |

---

## 6. Methodology

### 6.1 Data Preparation
- Removed empty columns and standardized features using `StandardScaler`.  
- Filtered out Class 7 as required by assignment instructions.  
- Remapped labels from `[1–5]` to `[0–4]` for XGBoost compatibility.

### 6.2 Model Training and Hyperparameter Tuning
- Used `GridSearchCV` with 3-fold cross-validation for parameter optimization.  
- Tuned key parameters for each model to ensure fair comparison.  
- Retained best parameters using cross-validation accuracy as selection criteria.

### 6.3 Evaluation Metrics
- Accuracy  
- Weighted F1-Score  
- ROC-AUC (macro-average)  
- Average Precision (macro-average)

### 6.4 Visualization and Analysis
- Accuracy and F1-Score bar charts  
- Multi-class ROC and PRC plots  
- Hyperparameter sensitivity graphs  
- Per-class ROC-AUC comparison  
- Confusion matrix (raw and normalized)  

---

## 7. Results Summary

| Model | Accuracy | Weighted F1 | ROC-AUC | Avg Precision |
|--------|-----------|-------------|----------|----------------|
| Random Forest | 0.939 | 0.938 | **0.995** | **0.980** |
| XGBoost | 0.935 | 0.934 | 0.994 | 0.975 |
| SVM | **0.943** | **0.942** | 0.993 | 0.970 |
| KNN | 0.940 | 0.939 | 0.990 | 0.964 |
| Logistic Regression | 0.897 | 0.895 | 0.985 | 0.945 |
| Decision Tree | 0.883 | 0.883 | 0.973 | 0.896 |
| Naive Bayes | 0.844 | 0.848 | 0.970 | 0.899 |
| Dummy (Prior) | 0.301 | 0.140 | 0.500 | 0.200 |

---

## 8. Key Findings

- **Best Overall Model:** Random Forest (ROC-AUC = 0.995, AP = 0.980)  
- **Close Competitors:** SVM and XGBoost also showed strong performance.  
- **Easiest Class:** Cotton Crop (AUC = 0.9995)  
- **Hardest Class:** Damp Grey Soil (AUC = 0.9859)  
- **Baseline Check:** Dummy Classifier confirmed random prediction level.  
- Consistent ranking across Accuracy, ROC-AUC, and Average Precision metrics.

---

## 9. Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
