# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:50:07 2022

@author: HP
"""


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv(r'Social_Network_Ads.csv')
print(df)
print(df.info())

#plt.scatter(df['Age'],df['Purchased'])
#plt.xlabel("age")
#plt.ylabel("purched")

train_x,test_x,train_y,test_y=train_test_split(df['Age'],df['Purchased'])

lr=linear_model.LogisticRegression()

lr.fit(train_x.values.reshape(-1,1),train_y.values.reshape(-1,1).ravel());

y_pred=lr.predict(test_x.values.reshape(-1,1))

print("testin y valus")
print(test_y)
print("predicted y valus")
print(y_pred)

plt.scatter(test_x,test_y)

plt.scatter(test_x,y_pred,c="red")
plt.xlabel("age")
plt.ylabel("purchesed")
plt.show()

ac=accuracy_score(test_y,y_pred)*100
print("accuracy score is",ac)

tn,fp,fn,tp=confusion_matrix(test_y,y_pred).ravel()
print(tn)
print(fp)
print(fn)
print(tp)

print("accuracy is")
a=(tn+tp)*100/(tn+fp+fn+tp)
print(a)
e=fn+fp/(tn+fp+fn+tp)
print("error is")
print(e)

































'''
#plt.scatter(df['Age'],df['Purchased'])
print(df.info())
df_x=np.asarray(df[['Age','EstimatedSalary']]);
df_y=np.asarray(df['Purchased'])
print(df_x)
print(df_y)

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.20)

lr=linear_model.LogisticRegression()

lr.fit(x_train.values.reshape(-1,1),y_train.reshape(-1,1).ravel());

y_pred=lr.predict(x_train.values.reshape(-1, 1))

print("expected valeu")
print(y_train);
print("modeul predicted")
print(y_pred)

plt.clf()
plt.scatter(x_test,y_test)
plt.scatter(x_test,y_pred,c="red")
plt.xlabel("Age")
plt.ylabel("Purchased")
plt.show()'''

