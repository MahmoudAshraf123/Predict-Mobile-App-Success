import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from Preprocessing import *

################################################ Regression #######################################################################

# Load data
df = read_data('AppleStore_training.csv')

#dealing with missing values
df.fillna(method='ffill', inplace=True)

#extract features and target
data, X, Y = extract_XY(df, 'user_rating')

X = xr_preprocess(X, 7)
Y = yr_preprocess(Y)

#load models
pkl_file = open('reg_models.pkl', 'rb')
regression_models = pickle.load(pkl_file)
pkl_file.close()

print("----------- Regressing Models -----------")
print("#")

#multible linear regression
multiple_linear_regression = regression_models[0]
prediction = multiple_linear_regression.predict(X)
print('Multiple Linear Regression Mean Square Error = ', metrics.mean_squared_error(Y, prediction))
print('Multiple Linear Regression r2 score = ', metrics.r2_score(Y, prediction))
print("#")

#polynomial linear regression
poly_features = regression_models[1]
polynomial_regression = regression_models[2]
prediction = polynomial_regression.predict(poly_features.fit_transform(X))
print('Polynomial Regression Mean Square Error = ', metrics.mean_squared_error(Y, prediction))
print('Polynomial Regression r2 score = ', metrics.r2_score(Y, prediction))
print("#")

#lasso linear regression
lasso_regression = regression_models[3]
prediction = lasso_regression.predict(X)
print('Lasso Regression Mean Square Error = ', metrics.mean_squared_error(Y, prediction))
print('Lasso Regression r2 score = ', metrics.r2_score(Y, prediction))
print("#")


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

################################################ Classification #######################################################################

# Load data
df = read_data('AppleStore_training_classification.csv')

#dealing with missing values
df.fillna(method='ffill', inplace=True)

#extract features and target
data, X, Y = extract_XY(df, 'rate')

X = xc_preprocess(X, 6)
Y = yc_prerocesss(Y)

#load models
pkl_file = open('class_models.pkl', 'rb')
classification_models = pickle.load(pkl_file)
pkl_file.close()

print("----------- Classification Models -----------")
print("#")

#Logistic Regression
logistic_regression = classification_models[0]
accuracy = logistic_regression.score(X, Y)
print('Logistic Regression accuracy = ' + str(accuracy))
print("#")

#SVM
svm = classification_models[1]
accuracy = svm.score(X, Y)
print('SVM accuracy = ' + str(accuracy))
print("#")

#KNN
knn = classification_models[2]
accuracy = knn.score(X, Y)
print('KNN accuracy = ' + str(accuracy))
print("#")

#Decision Tree Classifier
tree_clf = classification_models[3]
accuracy = tree_clf.score(X, Y)
print('Decision Tree accuracy = ' + str(accuracy))
print("#")

