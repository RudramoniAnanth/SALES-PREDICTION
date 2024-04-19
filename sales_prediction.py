import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Opening a heart dataset
data = pd.read_csv("Advertising.csv")

# Segregating the dataset into target and response variable
# Implementing the featureset
features = data.iloc[:,0:-1]
# Implementing the label set
# Percentage of heart attack risk
labels = data.iloc[:,-1]

# Putting these into two variables
X = features
y = labels

# Splitting the entire dataset into training-validation and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a X and Y for storing the required feature values
X = X_train
y = y_train

# Converting the X into numpy for executing the for loop
X = X.to_numpy()
y = y.to_numpy()

# Working on the 80% training-validation set for model building
# Initialize KFold with k=5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_number = 1
train_r2_scores = []
train_mse_scores = []
test_r2_scores = []
test_mse_scores = []

for train_index, test_index in kf.split(X):
    print(f"                 Fold {fold_number}                  ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    X_train, X_test1 = X[train_index], X[test_index]
    y_train, y_test1 = y[train_index], y[test_index]

    # Train a simple linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    print()
    print(" FOR TRAINING DATA : ")
    print()

    # Get the coefficients
    slope = lin_reg.coef_[0]
    intercept = lin_reg.intercept_

    # Make predictions on the training set
    y_pred_train = lin_reg.predict(X_train)

    # Calculate Mean Squared Error (MSE) for training
    mse_train = mean_squared_error(y_train, y_pred_train)
    train_mse_scores.append(mse_train)

    # Calculating R Squared from the equation
    rsquared_train = r2_score(y_train, y_pred_train)
    train_r2_scores.append(rsquared_train)

    # Print the coefficients, MSE, and R Squared
    print("Mean Squared Error (MSE):", mse_train)
    print("R Squared:", rsquared_train)
    print("slope", slope)
    print("intercept", intercept)

    print()
    print("FOR TESTING DATA : ")
    print()

    # Make predictions on the test set
    y_pred_test = lin_reg.predict(X_test1)

    # Calculate Mean Squared Error (MSE) for testing
    mse_test = mean_squared_error(y_test1, y_pred_test)
    test_mse_scores.append(mse_test)

    # Calculating R Squared from the equation
    rsquared_test = r2_score(y_test1, y_pred_test)
    test_r2_scores.append(rsquared_test)

    # Print the coefficients, MSE, and R Squared
    print("Mean Squared Error (MSE):", mse_test)
    print("R Squared:", rsquared_test)
    print("slope", slope)
    print("intercept", intercept)

    print()
    fold_number += 1
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Calculate the average R-squared and MSE scores
mean_train_r2 = np.mean(train_r2_scores)
mean_test_r2 = np.mean(test_r2_scores)
mean_train_mse = np.mean(train_mse_scores)
mean_test_mse = np.mean(test_mse_scores)
print("Over all average of all the metrics : ")
print()
print("Average Training R-squared:", mean_train_r2)
print("Average Testing R-squared:", mean_test_r2)
print("Average Training MSE:", mean_train_mse)
print("Average Testing MSE:", mean_test_mse)