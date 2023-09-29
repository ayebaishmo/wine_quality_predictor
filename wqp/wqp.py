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
de =  df.describe().T
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


df['best quality'] = [1 if x >5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40)

p = xtrain.shape, xtest.shape
print(p)

norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

models = [LogisticRegression() ,XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
    models[i].fit(xtrain, ytrain)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuaracy : ', metrics.roc_auc_score(
        ytest, models[i].predict(xtest)))
    print()