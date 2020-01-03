import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

df = pd.read_csv(r'../demo/titanic.csv')
#code start for boxplot and heatmap vvvv
sns.set_style("whitegrid")
sns.countplot(x="Survived",data=df)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=df)
#code end for boxplot and heatmap ^^^^

def compute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if(pd.isnull(Age)):
        if Pclass ==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

df['Age']=df[['Age','Pclass']].apply(compute_age,axis=1)
df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
#code start for getdummiesvvvv
pd.get_dummies(df['Sex'])
sex=pd.get_dummies(df['Sex'],drop_first=True) #none will be dropped because only two type of values
embark=pd.get_dummies(df['Embarked'],drop_first=True) # out of s,c,q; c will be dropped
Pcl=pd.get_dummies(df['Pclass'],drop_first=True) # out of 1,2,3; 1 will be dropped
df=pd.concat([df,sex,embark,Pcl],axis=1)
#code ends for getdummies^^^^

df.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
x=df.drop("Survived",axis=1)#copying all except survived from df
y=df["Survived"]##copying only survived from df
#code start for sklearn vvvv
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)#random_statte=0 means single variety of output

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report
classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)
#code ends for sklearn^^^^