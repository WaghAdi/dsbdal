# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:08:13 2022

@author: HP
"""

import pandas as pd
import numpy as np
df=pd.read_csv(r'student_marks.csv')
df.info()
print(df.head())
print(df.isnull())
#df=df.fillna(0)
print(df.head())
df['reading score']=df['math score'].fillna(df['math score'].mean())
print("after updation")
print(df.head())

#detecting outlayer

from scipy import stats
z=np.abs(stats.zscore(df['reading score']))
print(z)

sample=np.where(z<0.18)
ndf=df
for i in sample:
    ndf.drop(i,inplace=True)
print(ndf)