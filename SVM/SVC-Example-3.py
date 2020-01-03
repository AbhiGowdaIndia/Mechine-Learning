import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#load dataset from csv
df=pd.read_csv("daily_weather.csv")

#data cleaning
df.drop('number',axis=1,inplace=True)
df.dropna(inplace=True)
df['result']=((df['relative_humidity_3pm'])>24.99)*1

#split data into dependent and independent values
y=df['result']
x=df.drop(['result','relative_humidity_3pm'],axis=1)
# splitting dataset into training and testing sets
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)

# fit Logistic Regression to training dataset

model=SVC(kernel="linear",random_state=0)
model.fit(X_train,y_train)

# predicting result with testing datasets
y_pred=model.predict(x_test)

print("classification_report : \n",classification_report(y_test,y_pred))

print("confusion_matrix : \n",confusion_matrix(y_test,y_pred))

print("accuracy_score : ",accuracy_score(y_test,y_pred))


