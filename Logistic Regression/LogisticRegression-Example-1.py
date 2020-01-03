#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read or load Dataset
titanic_data=pd.read_csv("titanic_passenger_data.csv")

#Analyzing and Convert categorical variable into dummy/indicator variables
Pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
Sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)

#Reshaping the dataset
titanic_data=pd.concat([titanic_data,Pcl,Sex,embark],axis=1)
titanic_data.drop(['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
titanic_data.dropna(inplace=True)

#split data into x and Y axis
x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']

#Devide the dataset into training and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

#Train the model
from sklearn.linear_model import LogisticRegression
LogisticReg=LogisticRegression()
LogisticReg.fit(x_train,y_train)


#Predict values
y_predict=LogisticReg.predict(x_test)

#Calculating metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Classification report : \n",classification_report(y_test,y_predict))
print("Confusion matrix :\n",confusion_matrix(y_test,y_predict))
print("Accuracy score :",accuracy_score(y_test,y_predict))

