import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#Train the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

#Predict the values
pred = knn.predict(x_test)

#Calculating the error rate
error_rate = []
# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

#Ploating the graph for error rate  
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
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
print('WITH K=23')
print('\n')
print("Classification report : \n",classification_report(y_test,y_predict))
print("Confusion matrix :\n",confusion_matrix(y_test,y_predict))
print("Accuracy score :",accuracy_score(y_test,y_predict))