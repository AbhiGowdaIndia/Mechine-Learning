#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
#predict values using Linear regression
y_pred=lin_reg.predict(x_test)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Predict values using polynomial regression
y_poly_pred=lin_reg_2.predict(poly_reg.fit_transform(x_test))



# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print("Metrics for Linear regression \n")
print("-------------------------------")
print("Mean_squared_error = ",mean_squared_error(y_test,y_pred))
print("mean_absolute_error = ",mean_absolute_error(y_test, y_pred))
print("explained_variance_score = ",explained_variance_score(y_test, y_pred))
print("r2 score = ",r2_score(y_test, y_pred))

print("\n Metrics for ploynomial regression regression \n")
print("-------------------------------")
print("Mean_squared_error = ",mean_squared_error(y_test,y_poly_pred))
print("mean_absolute_error = ",mean_absolute_error(y_test, y_poly_pred))
print("explained_variance_score = ",explained_variance_score(y_test, y_poly_pred))
print("r2 score = ",r2_score(y_test, y_poly_pred))
