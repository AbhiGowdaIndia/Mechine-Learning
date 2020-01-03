# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

#Devide the dataset into training and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Train the model
from sklearn.linear_model import LinearRegression
SimpleLinearRegression=LinearRegression()
SimpleLinearRegression.fit(x_train,y_train)

#Predict values
y_predict=SimpleLinearRegression.predict(x_test)

#Plot the graph
plt.scatter(x_train, y_train, color = 'red')
plt.scatter(x_test,y_test,color='k')
plt.plot(x_train, SimpleLinearRegression.predict(x_train), color = 'blue')
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,explained_variance_score

print("Mean_squared_error = ",mean_squared_error(y_test,y_predict))
print("mean_absolute_error = ",mean_absolute_error(y_test, y_predict))
print("explained_variance_score = ",explained_variance_score(y_test, y_predict))
