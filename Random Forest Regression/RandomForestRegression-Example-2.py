import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
import csv
import operator
from sklearn.utils import shuffle
import pandas as pd
#Getting data from csv file
filename = 'train_data.csv'
dataset = pd.read_csv(filename)

#to shuffle data
dataset=shuffle(dataset)

#shortnenig data for clearifiction
#dataset = dataset[:100]


x = dataset.iloc[:,[0]].values
y = dataset.iloc[:,[1]].values
#plt.scatter(x,y,color='r',s=30,label='points')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
plt.scatter(x_train,y_train,color='r',s=10,label='points')
plt.scatter(x_test,y_test,color='g',s=10,label='points')
reg=RandomForestRegressor()
reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
plt.scatter(x_test,y_predict,color='c',s=10,label='points')
##regression
#from sklearn.ensemble import RandomForestRegressor
#
#reg = RandomForestRegressor()
#reg.fit(x,y)
#y_predict = reg.predict(x)
#
#sort_axis = operator.itemgetter(0)
#sorted_zip = sorted(zip(x,y_predict), key=sort_axis)
#x,y_predict= zip(*sorted_zip)
#
#plt.plot(x,y_predict,color='b',label='Regression line')
#
#plt.title("Random Forest regression2")
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.legend()
#plt.show()
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_test,y_predict), key=sort_axis)
#sorted_zip = zip(x_test,y_predict)
x_test_1,y_predict_1= zip(*sorted_zip)

plt.plot(x_test_1,y_predict_1,color='b',label='Regression line')

plt.title("Random Forest regression2")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
evs = explained_variance_score(y_test, y_predict)
m_score=reg.score(y_test,y_predict)
print(mse," ",mae," ",evs," ",m_score) 






