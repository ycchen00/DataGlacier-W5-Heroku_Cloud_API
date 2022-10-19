# Importing the libraries
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


bank = pd.read_csv("bank_data_sk.csv")

nbank = bank.drop('CustomerID', axis=1)

nbank = nbank.dropna()

# one_hot_encoded=pd.get_dummies(nbank.State)
# nbank=pd.concat([nbank,one_hot_encoded],axis=1)

train_set, test_set = train_test_split(nbank, train_size=0.8)

cols = ['Age', 'Balance', 'IsActiveMember', 'CheckingAcct']
x_cols = ['Age', 'Balance', 'IsActiveMember']
y_col = 'CheckingAcct'

train_set = train_set[cols]
test_set = test_set[cols]

# nprmalization
ncolumns = ['Age', 'Balance']

scaler = preprocessing.MinMaxScaler()

nbank[ncolumns] = scaler.fit_transform(nbank[ncolumns])

X_train, X_test, y_train, y_test = train_test_split(nbank[x_cols], nbank.CheckingAcct, train_size=0.8)

lr = LogisticRegression().fit(X_train, y_train)

# Saving model to disk
pickle.dump(lr, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))