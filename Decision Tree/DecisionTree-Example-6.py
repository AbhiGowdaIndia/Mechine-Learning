
#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split

#load dataset from csv file
dataset=pd.read_csv("SocialNetworkAds.csv")

# extracting independent variables
x = dataset.iloc[:,[2,3]].values

# extracting dependent variables
y= dataset.iloc[:,4].values

#Devide the dataset into training and test dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
x_train = standard_Scaler.fit_transform(x_train)
x_test = standard_Scaler.transform(x_test)

#Get the model object and providing the criterion
Dclassifier=DecisionTreeClassifier(criterion="entropy")

#train the model
Dclassifier.fit(x_train,y_train)

#predict the Values
y_predict=Dclassifier.predict(x_test)


print("Classification report :\n",classification_report(y_test,y_predict))

print("Confusion matrix : \n",confusion_matrix(y_test,y_predict))

print("Accuracy score :",accuracy_score(y_test,y_predict))

