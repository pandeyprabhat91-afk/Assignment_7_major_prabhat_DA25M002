DA5401 Assignment 7: Multi-Class Model Selection using ROC and Precision-Recall Curves

Author: Major Prabhat Pandey (DA25M002)
Program: M.Tech in Artificial Intelligence and Data Science
Date: October 2025

Project Overview

This project focuses on multi-class model selection and evaluation using ROC (Receiver Operating Characteristic) and Precision-Recall curves on the UCI Landsat Satellite dataset.
The objective is to compare multiple classification models and identify the best-performing one based on multiple evaluation metrics.

Objectives

Apply ROC and Precision-Recall Curves for multi-class classification using the One-vs-Rest strategy.

Perform hyperparameter tuning using GridSearchCV with 3-fold cross-validation.

Evaluate models using Accuracy, Weighted F1-Score, ROC-AUC, and Average Precision.

Conduct hyperparameter sensitivity analysis and per-class ROC-AUC evaluation.

Recommend the best-performing model based on comprehensive analysis.

Dataset

Source: UCI Statlog (Landsat Satellite) Dataset

Training samples: 4435

Test samples: 2000

Features: 36

Classes: 6 (Class 7 excluded as per assignment requirements)

After filtering Class 7, the dataset includes Classes 1–5 representing distinct land-cover types.

Models Implemented
Model	Description
KNN	k-Nearest Neighbors
Decision Tree	Entropy-based tree classifier
Logistic Regression	Multinomial with L2 regularization
SVM	RBF kernel with probability=True
Random Forest	Ensemble of decision trees
Naive Bayes	Gaussian Naive Bayes
XGBoost	Gradient Boosting Classifier
Dummy (Prior)	Baseline model predicting most frequent class
Development Timeline
Date	Activity
17 Oct 2025	Data loading, cleaning, and preprocessing
18 Oct 2025	GridSearchCV tuning for all models
19 Oct 2025	ROC computation using One-vs-Rest
20 Oct 2025	Precision-Recall curve computation and averaging fixes
21 Oct 2025	Final visualizations and confusion matrix analysis
Methodology
1. Data Preparation

Removed empty columns and standardized features using StandardScaler.

Filtered out Class 7.

Remapped labels from [1–5] to [0–4] for compatibility with XGBoost.

2. Model Training and Hyperparameter Tuning

Used GridSearchCV with 3-fold cross-validation.

Tuned parameters for KNN, Decision Tree, Logistic Regression, SVM, Random Forest, and XGBoost.

Best hyperparameters selected automatically.

3. Evaluation Metrics

Accuracy

Weighted F1-Score

ROC-AUC (macro-average)

Average Precision (macro-average)

4. Visualization and Analysis

Accuracy and F1 comparison plots

ROC and PRC for multi-class models

Hyperparameter sensitivity plots

Per-class ROC-AUC comparison

Confusion matrix heatmap

Results Summary
Model	Accuracy	Weighted F1	ROC-AUC	Avg Precision
Random Forest	0.939	0.938	0.995	0.980
XGBoost	0.935	0.934	0.994	0.975
SVM	0.943	0.942	0.993	0.970
KNN	0.940	0.939	0.990	0.964
Logistic Regression	0.897	0.895	0.985	0.945
Decision Tree	0.883	0.883	0.973	0.896
Naive Bayes	0.844	0.848	0.970	0.899
Dummy (Prior)	0.301	0.140	0.500	0.200
Key Findings

Best overall model: Random Forest (AUC = 0.995, AP = 0.980).

SVM and XGBoost are close competitors.

Dummy Classifier acts as the random baseline.

Cotton Crop was the easiest class to classify (AUC = 0.9995).

Damp Grey Soil was the hardest (AUC = 0.9859).

Consistency observed across ROC-AUC and Average Precision rankings.
