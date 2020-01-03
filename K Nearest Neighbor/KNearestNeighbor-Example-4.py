#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
%matplotlib inline

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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
x_train = standard_Scaler.fit_transform(x_train)
x_test = standard_Scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)

error_rate = []
# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)

print('WITH K=1')
print('\n')
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Classification report : \n",classification_report(y_test,y_predict))
print("Confusion matrix :\n",confusion_matrix(y_test,y_predict))
print("Accuracy score :",accuracy_score(y_test,y_predict))

# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)

print('WITH K=23')
print('\n')
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Classification report : \n",classification_report(y_test,y_predict))
print("Confusion matrix :\n",confusion_matrix(y_test,y_predict))
print("Accuracy score :",accuracy_score(y_test,y_predict))


