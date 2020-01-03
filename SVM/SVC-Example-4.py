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
model=SVC(kernel="linear",random_state=0)
model.fit(x_train,y_train)

# predicting result with testing datasets
y_pred=model.predict(x_test)

print("classification_report : \n",classification_report(y_test,y_pred))

print("confusion_matrix : \n",confusion_matrix(y_test,y_pred))

print("accuracy_score : ",accuracy_score(y_test,y_pred))


