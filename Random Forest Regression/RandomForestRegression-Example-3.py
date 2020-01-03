# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read Dataset
ds=pd.read_csv("salary.csv")

#split data into x and Y axis
x=ds.iloc[:,:-1].values
y=ds.iloc[:,1].values

#Devide the dataset into training and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# Predicting a test result
y_pred = regressor.predict(x_test)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_train, y_train, color = 'green')
plt.scatter(x_test, y_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,explained_variance_score,r2_score

print("Mean_squared_error = ",mean_squared_error(y_test,y_pred))
print("mean_absolute_error = ",mean_absolute_error(y_test, y_pred))
print("explained_variance_score = ",explained_variance_score(y_test, y_pred))
print("r2_score = ",r2_score(y_test, y_pred))

