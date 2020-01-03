#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load Dataset from CSV
df = pd.read_csv("Classified_Data.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))

x= scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(x,columns=df.columns[:-1])
df_feat.head()
y=df['TARGET CLASS']

from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)

x_train, x_test, y_train, y_test = train_test_split(x,df['TARGET CLASS'],test_size=0.30)

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

