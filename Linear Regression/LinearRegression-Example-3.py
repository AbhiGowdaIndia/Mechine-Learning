#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read Dataset
ds=pd.read_csv("train_data.csv")

#split data into x and Y axis
x=ds.iloc[:,:-1].values
y=ds.iloc[:,1].values

#Devide the dataset into training and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/4,random_state=0)

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
