# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:26:54 2022

@author: HP
"""

import pandas as pd
import seaborn as sns
df=sns.load_dataset('titanic')
print(df.head())
print(df.info())

sns.boxplot(df['age'],df['sex'])
sns.boxplot(df['age'],df['sex'],df['survived'])