import pandas
import numpy
from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB

# Loading dataset from csv
dataset=pd.read_csv("SocialNetworkAds.csv")

# extracting independent variables
X = dataset.iloc[:,[2,3]].values

# extracting dependent variables
Y = dataset.iloc[:,4].values

# splitting dataset into training and testing sets
X_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

# Feature scaling
standard_Scaler=StandardScaler()
X_train = standard_Scaler.fit_transform(X_train)
x_test = standard_Scaler.transform(x_test)

#Train the model
model=GaussianNB()
model.fit(X_train,y_train)

#Predict the values
y_predict=model.predict(x_test)

#Calculate the metrics
print("Classification report : \n",classification_report(y_test,y_predict))

print("Confusion matrix : \n",confusion_matrix(y_test,y_predict))

print("Accuracy score : ",accuracy_score(y_test,y_predict))
