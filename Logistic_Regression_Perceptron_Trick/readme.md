# Logistic Regression using the Perceptron Trick

This notebook provides a from-scratch implementation of a Logistic Regression classifier using Python and NumPy. Instead of using standard batch Gradient Descent, this implementation uses a stochastic update rule often referred to as the **Logistic Perceptron Trick** to learn the optimal decision boundary.

## 📌 Overview
The goal of this notebook is to build an intuitive understanding of how linear classifiers update their weights point-by-point based on prediction errors. 

The notebook walks through:
1. **Mathematical Foundations**: Defining the Sigmoid function to map continuous outputs to probabilities (0 to 1).
2. **The Update Rule**: Implementing the Perceptron trick to adjust weights dynamically based on the error of individual data points.
3. **Data Generation**: Creating a synthetic, linearly separable 2D dataset using `scikit-learn`.
4. **Training & Evaluation**: Training the custom model to find the optimal weights and bias.
5. **Visualization**: Calculating and plotting the mathematical decision boundary separating the classes.

## 🛠️ Dependencies
To run this notebook, you will need the following Python libraries:
* `numpy` (for matrix operations and math)
* `matplotlib` (for plotting the data and decision boundary)
* `scikit-learn` (specifically `make_classification` for generating the synthetic dataset)

## 🚀 Usage
1. Open `Logistic_Regression_Perceptron_Trick.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells sequentially.
3. The notebook will generate a synthetic dataset, train the custom logistic regression model over 1,000 epochs, and output the learned weights (Intercept, W1, W2).
4. The final cell will plot a 2D scatter plot showing the two classes and the calculated green decision boundary line.

## 📊 Results
On the synthetic linearly separable dataset, the custom classifier successfully converges, achieving **100% Training Accuracy**. The final visualization clearly demonstrates how the mathematical formulation $w_0 + w_1x_1 + w_2x_2 = 0$ perfectly separates the two generated clusters.
