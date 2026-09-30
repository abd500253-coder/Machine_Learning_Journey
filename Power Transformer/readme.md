# Power Transformations to Improve Linear Models

This repository contains a Jupyter Notebook demonstrating how to use the `PowerTransformer` from `scikit-learn` to apply **Box-Cox** and **Yeo-Johnson** transformations. 

The primary goal of this project is to transform non-normally distributed features into a more Gaussian-like (normal) distribution. This process helps stabilize variance, minimizes skewness, and ultimately improves the predictive performance of linear models like **Linear Regression**.

## 📊 Dataset
The notebook uses the **Concrete Compressive Strength** dataset. 
* **Features:** Various ingredients used to make concrete (e.g., Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, etc.).
* **Target (`y`):** Concrete Compressive Strength.
* *Note: The dataset contains values equal to zero in columns like Blast Furnace Slag and Fly Ash, which poses a specific challenge for certain mathematical transformations.*

## 🛠️ Libraries & Tools Used
* `pandas` & `numpy` (Data manipulation)
* `seaborn` & `matplotlib.pyplot` (Data visualization)
* `scipy.stats` (QQ plots and probability distributions)
* `scikit-learn` (Model building, cross-validation, and preprocessing)

## 🚀 Key Workflow
1. **Exploratory Data Analysis (EDA):** Loading the dataset and checking for missing values and statistical properties.
2. **Baseline Model:** Training a baseline `LinearRegression` model on the raw, untransformed data to establish a benchmark $R^2$ and Cross-Validation score.
3. **Visualizing Original Distributions:** Using Histograms and QQ-Plots to inspect the skewness of the original features.
4. **Box-Cox Transformation:** 
   * *Challenge:* Box-Cox requires strictly positive values ($>0$).
   * *Solution:* A small constant (`0.000001`) is added to the features to bypass zero values before applying the transformation.
5. **Yeo-Johnson Transformation:** Applying the default Yeo-Johnson method, which natively handles positive, zero, and negative values.
6. **Performance & Lambda Comparison:** Visualizing the transformed distributions and comparing the optimal $\lambda$ (lambda) values chosen by both transformation methods.

## 📈 Results Overview
The notebook compares the **$R^2$ Score** and **Cross-Validation Score** across three states:
1. Baseline (No Transformation)
2. Post Box-Cox Transformation
3. Post Yeo-Johnson Transformation

By comparing the distribution plots (Before vs. After) and the model scores, we can observe how making the data more Gaussian-like directly impacts the accuracy of the Linear Regression model.
