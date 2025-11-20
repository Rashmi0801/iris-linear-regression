"""iris_regression.py

Script version of the notebook. No outputs stored here — just code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    irisset = datasets.load_iris()
    df = pd.DataFrame(irisset.data, columns=irisset.feature_names)
    df['target'] = irisset.target
    df['target_name'] = df['target'].map(dict(enumerate(irisset.target_names)))
    
    # Use first 50 samples (Iris setosa)
    X = irisset.data[:50, 0:1]   # Sepal length
    y = irisset.data[:50, 1]     # Sepal width
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = LinearRegression().fit(X_train, y_train)
    
    y_pred_test = reg.predict(X_test)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print('Test MSE =', round(mse_test, 4))
    print('Test R2 =', round(r2_test, 4))

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    minv, maxv = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle='--')
    plt.xlabel('Actual Sepal Width (cm)')
    plt.ylabel('Predicted Sepal Width (cm)')
    plt.title('Actual vs Predicted (Test set)')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
