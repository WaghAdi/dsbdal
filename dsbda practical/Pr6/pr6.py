# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:25:40 2022

@author: HP
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score



df=pd.read_csv(r'iris.csv')
df.info()
#print(df.isnull())
#df_x=df.iloc[:,:4].values
#df_y=df[''] 

df_x=df.iloc[:,:4].values
df_y=df['Species']

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y)

nb=GaussianNB()
nb.fit(x_train,y_train)

y_pred=nb.predict(x_test)
print("test y")
print(y_test)
print("predicted y")
print(y_pred)

#tn,fp,fn,tp=confusion_matrix(y_test,y_pred)
#total=tn+fpfn+tp

#accuracy=(tn+tp)*100/total
#print(accuracy)
e=accuracy_score(y_test,y_pred)
print("accuracy")
print(e)













































