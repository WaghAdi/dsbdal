# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:12:28 2022

@author: HP
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset('titanic')
df.info()
print(df.head())

#sns.distplot(df['fare'])
#sns.rugplot(df['fare'])

#sns.pairplot(df)

#sns.jointplot(x=df['age'],y=df['fare'],data=df)

#print(sns.histplot(x=df['age'],y=df['fare'],data=df))


