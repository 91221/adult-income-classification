Adult Income Classification

A. PROBLEM STATEMENT

The objective of this project is to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

This is a binary classification problem where:

0 → Income ≤ 50K

1 → Income > 50K

The goal is to compare multiple machine learning models and evaluate their performance using standard classification metrics.

B. DATASET DESCRIPTION

The dataset used is the Adult Income Dataset (Census Income dataset).

It contains demographic and work-related features such as:

Age

Workclass

Education

Marital Status

Occupation

Relationship

Race

Sex

Capital Gain

Capital Loss

Hours per week

Native country

Target variable:

Income (<=50K or >50K)

The dataset contains both numerical and categorical features.
Categorical variables were encoded before training the models.

C. MODELS USED

The following six machine learning models were implemented and evaluated:

Logistic Regression

Decision Tree

k-Nearest Neighbors (kNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

MODEL COMPARISION TABLE
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|------------|---------|----------|------|
| Logistic Regression | 0.802752 | 0.810695 | 0.678899 | 0.394141 | 0.498736 | 0.408691 |
| Decision Tree | 0.806564 | 0.740368 | 0.612190 | 0.608522 | 0.610351 | 0.481704 |
| kNN | 0.770927 | 0.663644 | 0.570588 | 0.322903 | 0.412415 | 0.301210 |
| Naive Bayes | 0.786508 | 0.828922 | 0.656891 | 0.298269 | 0.410256 | 0.336790 |
| Random Forest (Ensemble) | 0.852478 | 0.912829 | 0.798828 | 0.544607 | 0.647664 | 0.574951 |
| XGBoost (Ensemble) | 0.865407 | 0.924381 | 0.774682 | 0.647803 | 0.705584 | 0.623406 |

OBSERVATIONS ON MODEL PERFORMANCE

Logistic Regression:

Logistic Regression performed well as a baseline model. It achieved stable accuracy and good AUC, indicating partial linear separability in the dataset. However, recall was comparatively lower, meaning some high-income individuals were misclassified.

Decision Tree:

The Decision Tree model captured non-linear relationships effectively but showed slight overfitting. While recall improved compared to Logistic Regression, overall generalization performance was slightly lower.

k-Nearest Neighbors (kNN):

kNN produced balanced results with moderate performance across all metrics. However, it is sensitive to feature scaling and computationally expensive during prediction.

Naive Bayes:

Naive Bayes was computationally efficient but achieved lower performance due to its strong independence assumption between features.

Random Forest (Ensemble):

Random Forest significantly improved performance by reducing variance and preventing overfitting. It achieved higher precision and F1-score, demonstrating better balance between false positives and false negatives.

XGBoost (Ensemble):

XGBoost achieved the best overall performance across all metrics. The high AUC and MCC values indicate strong class separability and effective handling of class imbalance. It demonstrated superior generalization capability compared to other models.

FINAL ANALYTICAL CONCLUSION 

Ensemble methods (Random Forest and XGBoost) clearly outperformed individual models in terms of Accuracy, AUC, F1 Score, and MCC.

XGBoost achieved the best overall performance due to its boosting mechanism, which sequentially reduces errors and optimizes model performance.

The Matthews Correlation Coefficient (MCC) confirms that ensemble models handled class imbalance more effectively compared to simpler models.

Therefore, XGBoost is selected as the final recommended model for the Adult Income classification task.
