import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split

#load dataset from csv
df=pd.read_csv("daily_weather.csv")

#data cleaning
df.drop('number',axis=1,inplace=True)
df.dropna(inplace=True)
df['result']=((df['relative_humidity_3pm'])>24.99)*1

#split data into dependent and independent values
y=df['result']
x=df.drop(['result','relative_humidity_3pm'],axis=1)

#Devide the dataset into training and test dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

#Train the model
Dtree=DecisionTreeClassifier(criterion="entropy")
Dtree.fit(x_train,y_train)

#predict the values
Dt_y_pred=Dtree.predict(x_test)

print("Classification report :\n",classification_report(y_test,Dt_y_pred))

print("Confusion matrix : \n",confusion_matrix(y_test,Dt_y_pred))

print("Accuracy score :",accuracy_score(y_test,Dt_y_pred)*100)
