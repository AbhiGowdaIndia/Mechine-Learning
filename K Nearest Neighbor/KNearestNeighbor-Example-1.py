import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

df = pd.read_csv("Classified_Data.csv")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS',axis=1))

x= scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(x,columns=df.columns[:-1])
df_feat.head()
#x=df.drop('TARGET CLASS',axis=1)
y=df['TARGET CLASS']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,df['TARGET CLASS'],test_size=0.30,random_state=0)

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
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)

print('WITH K=23')
print('\n')
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Classification report : \n",classification_report(y_test,y_predict))
print("Confusion matrix :\n",confusion_matrix(y_test,y_predict))
print("Accuracy score :",accuracy_score(y_test,y_predict))


