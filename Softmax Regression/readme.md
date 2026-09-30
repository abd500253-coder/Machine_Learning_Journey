# Softmax Regression (Multinomial Logistic Regression)

This repository contains a Jupyter Notebook demonstrating the implementation and application of Softmax Regression, a generalization of logistic regression used for multi-class classification tasks. 

Unlike standard logistic regression which returns a single probability for a binary outcome, Softmax Regression outputs a probability distribution across $K$ possible, mutually exclusive classes.

## 📓 Notebook Overview

The notebook is divided into two main sections:

### 1. The Softmax Function from Scratch
This section covers the mathematical foundation of the softmax function. It takes a vector of raw scores (logits) and normalizes it into a probability distribution.
* **Implementation:** Built from scratch using `NumPy`.
* **Formula:** $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$
* **Demonstration:** Shows how raw scores like `[2, 5, -1]` are converted into normalized probabilities summing to 1.

### 2. Softmax Regression using Scikit-Learn
This section demonstrates how to apply Softmax Regression using the `scikit-learn` library on a real-world dataset.
* **Dataset:** The classic **Iris dataset** (3 different classes of iris flowers).
* **Model:** `LogisticRegression(multi_class='multinomial', max_iter=200)`
* **Key Methods Explored:**
  * `.predict_proba()`: Outputs an array where each row sums to 1 (the softmax probabilities).
  * `np.argmax()`: Extracts the final class predictions by selecting the index with the highest probability.

## 🛠️ Requirements & Dependencies

To run this notebook locally or on Google Colab, you will need the following Python libraries:
* `numpy`
* `scikit-learn`

You can install the required dependencies using pip:
```bash
pip install numpy scikit-learn
