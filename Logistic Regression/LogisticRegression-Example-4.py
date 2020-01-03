import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Loading dataset from csv
df=pd.read_csv("data.csv")

#split data into dependent and independent values
x=df.drop("gender",axis=1)
y=df["gender"]


# splitting dataset into training and testing sets
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)

# fit Logistic Regression to training dataset
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

# predicting result with testing datasets
y_pred=log_reg.predict(x_test)

print("Classification report :\n",classification_report(y_test,y_pred))

print("Confusion matrix : \n",confusion_matrix(y_test,y_pred))

print("Accuracy score :",accuracy_score(y_test,y_pred))
