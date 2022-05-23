# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:38:47 2022

@author: HP
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns



#sns.boxplot(df['Reading_score'],df['math_score'])
df=pd.read_csv(r'C:\Users\HP\Desktop\Adi\Irisflower.csv')
print(df.info())

#sns.histplot(df['SepalLengthCm'])
sns.boxplot(df['SepalLengthCm'])