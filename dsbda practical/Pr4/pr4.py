# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:53:45 2022

@author: HP
"""

import pandas as pd
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

boston=load_boston()
print(boston)
df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_y=pd.DataFrame(boston.target);
print(df_x)
print(df_y)

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y)

lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

print("y value in test are")
print(y_test)
print("y value in model are")
print(y_pred)

print("finding error ")
print(np.mean(y_pred-y_test)**2)