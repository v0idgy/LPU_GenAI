# Simple Linear Regression using Scikit-Learn and Matplotlib
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 1) * 10  # Random data for independent variable
y = 2.5 * X + np.random.randn(100, 1) * 2  # Dependent variable with some noise

model = LinearRegression()  # Create a linear regression model
model.fit(X, y)  # Fit the model to the data

X_fit = np.linspace(0, 10, 100).reshape(-1, 1)  # Generate points for the regression line
y_fit = model.predict(X_fit)  # Predict the target variable for the generated points

slope = model.coef_[0][0]  # Get the slope of the regression line
intercept = model.intercept_[0]  # Get the intercept of the regression line
y_pred = model.predict(X)  # Predict the target variable for the original data
r2 = r2_score(y, y_pred)  # Calculate the R-squared score
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, R-squared: {r2:.2f}")  # Print the slope, intercept, and R-squared score