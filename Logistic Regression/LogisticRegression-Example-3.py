import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns

# Loading dataset from csv
df=pd.read_csv("kyphosis_data.csv")

# extracting independent variables
#split dataset into dependent and independent variables
X=df.drop("Kyphosis",axis=1)
Y=df["Kyphosis"]

# splitting dataset into training and testing sets
X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)

# fit Logistic Regression to training dataset
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

# predicting result with testing datasets
y_pred=log_reg.predict(x_test)

#calculating metrics
print("Classification report :\n",classification_report(y_test,y_pred))

print("Confusion matrix : \n",confusion_matrix(y_test,y_pred))

print("Accuracy score :",accuracy_score(y_test,y_pred))
