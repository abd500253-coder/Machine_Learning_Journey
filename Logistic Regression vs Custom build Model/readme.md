# 🚀 Logistic Regression: Custom Built vs. Scikit-Learn

### 1. Project Title & What It Is

**A friendly summary:** This project builds a classification machine learning model completely from scratch and compares it against the industry standard to see how well we did!

**The core problem:** We are trying to predict a "Yes or No" outcome—specifically, whether a customer will "Churn" (leave a service) or "Stay." Instead of just using a black-box tool to do this, we are looking under the hood to write the math ourselves, helping us truly understand how a computer learns to make these predictions.

### 2. The Data We Used

Because we are focusing on the mechanics of the algorithm, we used a **synthetic (computer-generated) dataset**. We generated this directly in the code using a tool from Scikit-Learn.

It contains 2,000 examples (think of them as 2,000 imaginary customers) with 8 different "features" (metrics or traits). To make it realistic, 5 of these metrics strongly predict whether the customer stays or leaves, 2 are overlapping/redundant, and the data is split cleanly into our two categories: Churn or Stay.

### 3. Step-by-Step Workflow (The 'Why' Behind the Code)

* **Data Cleaning:** Because we generated the data ourselves, it was mostly perfectly clean! However, we did have to do a bit of mathematical preparation: we added a column of "1s" to our data matrix. This acts as a mathematical anchor (called the "intercept" or "bias") so our custom model can draw its prediction line accurately.
* **Exploring the Data:** We set the parameters of our data to have a "clean separation." This means the customers who churn and those who stay are distinct enough that we can clearly test if our custom-built math is actually finding the right patterns without getting confused by too much random noise.
* **Machine Learning Model:** We built a **Logistic Regression** model using a technique called *Gradient Descent*. In simple terms, Logistic Regression finds the best dividing line between two groups and calculates the *probability* (from 0 to 100%) that a new customer belongs to one group over the other. Gradient Descent is the learning process: the model makes a guess, checks how wrong it is (calculating the "Loss"), and takes a tiny step toward the correct answer, repeating this hundreds of times until it gets it right!

### 4. Libraries Needed (The Tools)

* **`numpy`**: The ultimate math and numbers toolbox for Python. We used this heavily to handle matrix multiplication, calculate our probabilities (the Sigmoid function), and update our model's learning weights.
* **`pandas`**: A fantastic library for organizing data into neat, readable tables. We used it at the very end to print a side-by-side comparison scorecard of our custom model vs. the professional model.
* **`scikit-learn` (`sklearn`)**: The go-to standard machine learning toolkit in Python. We used it to easily generate our mock dataset and to run the "official" Logistic Regression model so we had a gold standard to compare our homemade code against.

### 5. How to Run the Notebook

1. Open **Google Colab** in your web browser (it is completely free and requires zero setup or installation).
2. Go to **File > Upload notebook** and upload this `.ipynb` file.
3. Click on the first cell of code and press **`Shift + Enter`** (or click the round "Play" button on the left of the cell) to run it.
4. Keep pressing `Shift + Enter` to walk through the notebook step-by-step. Read the text blocks as you go, and watch the custom model learn right before your eyes!

---
