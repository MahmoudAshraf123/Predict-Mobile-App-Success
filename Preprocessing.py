from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

def read_data(file_name):
    df = pd.read_csv(file_name)
    # drop currency, ver columns
    del df['currency']
    del df['ver']
    return df

def extract_XY(df, target_column):
    data = df.iloc[:, :]
    X = data.iloc[:, 2:-1].copy()  # without id, track_name, currency and user_rating
    Y = data[target_column]
    return data, X, Y

#we use this to replace the categories with
def one_hot_encoding(X, n):
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [n])], remainder='passthrough')
    X = np.array(columnTransformer.fit_transform(X), dtype=np.str)
    return X

#we use this to delete the plus character
def delete_plus_in_rate(X):
    for i in range(len(X[:, -4])):
        length_without_plus = len(X[:, -4][i]) - 1
        X[:, -4][i] = X[:, -4][i][:length_without_plus]
    return X

def feature_scaling(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

def xr_preprocess(X , n):
    X = one_hot_encoding(X, n)

    X = delete_plus_in_rate(X)

    # convert the type to float
    X = X.astype(np.float_)

    return X

def yr_preprocess(Y):
    Y = np.asarray(Y)
    Y = Y.astype(np.float_)
    return Y

def xc_preprocess(X , n):
    X = one_hot_encoding(X, n)

    X = delete_plus_in_rate(X)

    X = feature_scaling(X)

    # convert the type to float
    X = X.astype(np.float_)

    return X


def yc_prerocesss(y):
    y = list(y)
    #yy = []
    for i in range(len(y)):
        if y[i] == 'High':
            y[i] = 1
        elif y[i] == 'Low':
            y[i] = -1
        else:
            y[i] = 0

    y = np.asarray(y)
    y = y.astype(np.float_)

    return y
