import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris=pd.read_csv("IRIS_data.csv")

x = iris.iloc[:, :-1]  # we only take the first two features. We could                     
                        # avoid this ugly slicing by using a two-dim dataset
Y = iris.iloc[:,4]

X_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=0)

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


