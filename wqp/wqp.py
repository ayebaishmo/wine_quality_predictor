import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('winequality.csv')
#print(df.head)
#df.info()
#de = df.describe().T
#print(de)
#print(df.isnull().sum())

for cole in df.columns:
    if df[cole].isnull().sum() > 0:
        df[cole] = df[cole].fillna(df[cole].mean())

#x = df.isnull().sum().sum()
#print(x)

#df.hist(bins=20, figsize=(10, 10))
#plt.show()


#plt.bar(df['quality'], df['alcohol'])
#plt.xlabel('quality')
#plt.ylabel('alcohol')
#plt.show()

df = df.drop('total sulfur dioxide', axis=1)

print(df.head)

