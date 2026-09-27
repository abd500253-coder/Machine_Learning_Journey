# Student Admission Prediction Pipeline

**Author:** Abdul Rehman Shafiq

## Overview
This repository contains a machine learning project focused on predicting student admissions using an end-to-end `scikit-learn` pipeline. The project demonstrates best practices for data preprocessing, avoiding data leakage, and evaluating multiple classification algorithms systematically.

## Features
* **Dynamic Preprocessing:** Utilizes `make_column_selector` to dynamically identify and separate numeric and categorical columns.
* **Robust Pipelines:** Implements `Pipeline` and `ColumnTransformer` to handle missing values, scale features, and encode categorical data in a single, streamlined workflow.
  * *Numeric Data:* Missing values imputed with the mean; scaled using `StandardScaler`.
  * *Categorical Data:* Missing values imputed with the most frequent category; encoded using `OneHotEncoder`.
* **Model Evaluation:** Trains and evaluates multiple classifiers, including:
  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier

## Dataset
The project relies on a dataset named `student_admission.csv`. It contains 30 records with the following features:
* **Numeric:** `Age`, `Study_Hours`, `Previous_Score`, `Family_Income`
* **Categorical:** `City`, `Gender`, `Internet_Access`
* **Target:** `Admission`

## Technologies Used
* Python 3
* Pandas
* NumPy
* Scikit-Learn

## How to Use
1. Clone the repository and ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`).
2. Place the `student_admission.csv` file in the same directory as the notebook (or update the file path in the code).
3. Open the `Column Transformer and Pipeline.ipynb` notebook in Jupyter or Google Colab.
4. Run all cells to load the data, execute the preprocessing pipeline, train the models, and view the accuracy scores and the interactive pipeline visualization.
