# Titanic Survival Prediction Pipeline

This repository contains a complete, end-to-end machine learning workflow implemented in a Jupyter Notebook (Google Colab). The project uses the classic Titanic dataset to predict passenger survival using a **Bernoulli Naive Bayes** classifier. It heavily features `scikit-learn`'s `Pipeline` and `ColumnTransformer` to ensure robust and reproducible data preprocessing, followed by hyperparameter tuning using `GridSearchCV`.

## 🚀 Project Overview

The notebook is structured into 12 clear, sequential steps that guide you through the entire machine learning lifecycle, from data loading to probabilistic predictions.

### Key Features

* **Feature Engineering:** Creates a new `with_family` feature based on existing `sibsp` (siblings/spouses) and `parch` (parents/children) columns.
* **Advanced Preprocessing:** Utilizes `ColumnTransformer` to apply different preprocessing steps to numeric, categorical, and binary features simultaneously.
* **Hyperparameter Tuning:** Implements cross-validated grid search to find the optimal parameters for both the preprocessing steps (e.g., imputation strategy) and the classifier.

## 📦 Dependencies

To run this notebook, you will need Python 3.x and the following libraries:

* `numpy`
* `pandas`
* `seaborn` (used for easily loading the Titanic dataset)
* `scikit-learn`

## 🛠️ Data Preprocessing Strategy

The data is split into a training set and a testing set (80/20 split) with stratified sampling to maintain class proportions. The preprocessing pipeline handles features as follows:

1. **Numeric Features (`age`, `fare`)**:
* Imputes missing values using the median (or mean, depending on the grid search results).
* Scales values using `StandardScaler`.


2. **Categorical Features (`embarked`)**:
* Imputes missing values using the most frequent value (or a constant).
* Applies `OneHotEncoder` (ignoring unknown categories).


3. **Binary Features (`pclass`, `with_family`)**:
* Imputes missing values using the most frequent value.
* Scales values.
* Converts to strict binary using `Binarizer`.



## 🏗️ Model Architecture

* **Estimator:** `BernoulliNB` (Bernoulli Naive Bayes).
* **Grid Search Parameters:**
* `classifier__alpha`: [0.01, 0.1, 0.5, 1.0, 5.0]
* `classifier__fit_prior`: [True, False]
* `preprocessor__numeric__imputer__strategy`: ['mean', 'median']
* `preprocessor__categorical__imputer__strategy`: ['most_frequent', 'constant']



## 📋 Notebook Workflow

The notebook is divided into the following sections:

1. **Import Libraries:** Loads required modules from pandas, numpy, seaborn, and sklearn.
2. **Load Dataset:** Loads the Titanic dataset, performs feature engineering, and drops redundant columns.
3. **Define Features and Target:** Separates the target variable (`survived`) from the input features and categorizes feature types.
4. **Train / Test Split:** Splits the data using `train_test_split`.
5. **Preprocessing Pipelines:** Defines individual pipelines for numerical, categorical, and binary data.
6. **Build Complete Pipeline:** Combines preprocessing and the `BernoulliNB` classifier.
7. **Train Baseline Model:** Trains the initial pipeline and evaluates it using accuracy, confusion matrix, and a classification report.
8. **Hyperparameter Tuning:** Uses `GridSearchCV` to test various parameter combinations across 5-fold cross-validation.
9. **Best Model:** Extracts the best-performing estimator and its optimal parameters.
10. **Final Test Evaluation:** Evaluates the tuned model on the unseen test set.
11. **Predict Class Probabilities:** Generates the probability of survival/non-survival for test instances.
12. **Compare Actual and Predicted Results:** Displays a side-by-side comparison of the model's predictions versus the actual ground truth.
