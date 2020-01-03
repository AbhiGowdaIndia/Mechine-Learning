#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score


#Read or load Dataset
boston=load_boston() 
bs=pd.DataFrame(boston.data)
bs['PRICE']=boston.target


#split data into x and Y axis
y=bs['PRICE']
x=bs.drop('PRICE',axis=1)


#Devide the dataset into training and test dataset
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=5)

#Data preprocessing
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)

#Train the model
model=LinearRegression()
model.fit(x_train,y_train)

#Predict values
y_predict=model.predict(x_test)

#calculating metrics
print("Mean_squared_error = ",mean_squared_error(y_test,y_predict))
print("mean_absolute_error = ",mean_absolute_error(y_test, y_predict))
print("explained_variance_score = ",explained_variance_score(y_test, y_predict))