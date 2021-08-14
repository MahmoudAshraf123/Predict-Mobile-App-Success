import pickle
from Preprocessing import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
import time

# Load data
df = read_data('AppleStore_training.csv')

#drop rows with missing data
df.dropna(axis=0, how='any', inplace=True)

#extract features and target
data, X, Y = extract_XY(df, 'user_rating')

#print(list(X)[7])

X = xr_preprocess(X, 7)
Y = yr_preprocess(Y)


corr = data.corr()
# Correlation training features with the Value
top_feature = corr.index[abs(corr['user_rating'] > 0)]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True)



models_list = []
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#multible linear reggresion
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
Start_time = time.time()
print("i've started now the multiple linear regression")
reg = LinearRegression().fit(X_train, y_train)
models_list.append(reg)

s = reg.score(X_test, y_test)
prediction = reg.predict(X_test)
print('Co-efficient of linear regression', reg.coef_)
print('Intercept of linear regression model', reg.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
true_rating_value = np.asarray(y_test)[0]
predicted_rating_value = prediction[0]
print('True value for the User rate: ' + str(true_rating_value))
print('Predicted value for the User rate: ' + str(predicted_rating_value))
end_time = time.time()
duration = end_time - Start_time
print('the time taken for training multiple linear reg. : ' + str(duration) + ' second')

plt.scatter(y_test, prediction)
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('prediction', fontsize=20)
plt.show()

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#polynomial linear regression
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

Start_time = time.time()
print("i've started now the polynomial linear regression")

poly_features = PolynomialFeatures(degree=4)
models_list.append(poly_features)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
models_list.append(poly_model)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Co-efficient of poly linear regression', poly_model.coef_)
print('Intercept of poly linear regression model', poly_model.intercept_)
print('Mean Square Error of poly ', metrics.mean_squared_error(y_test, prediction))
true_rating_value = np.asarray(y_test)[0]
predicted_rating_value = prediction[0]
print('True value for the success rate in the test set: ' + str(true_rating_value))
print('Predicted value for the success rate in the test set : ' + str(predicted_rating_value))


end_time = time.time()
duration = end_time - Start_time
print('the time taken for training polynomial linear reg. : ' + str(duration) + ' second')


plt.scatter(y_test, prediction)
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('prediction', fontsize=20)
plt.show()

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#lasso linear reggresion
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
Start_time = time.time()
print("i've started now the lasso linear regression")

reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, y_train)
models_list.append(reg)

prediction = reg.predict(X_test)
print('Co-efficient of lasso regression', reg.coef_)
print('Intercept of lasso regression model', reg.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
true_rating_value = np.asarray(y_test)[0]
predicted_rating_value = prediction[0]
print('True value for the success rate in the test set: ' + str(true_rating_value))
print('Predicted value for the success rate in the test set : ' + str(predicted_rating_value))

end_time = time.time()
duration = end_time - Start_time
print('the time taken for training lasso linear reg. : ' + str(duration) + ' second')


plt.scatter(y_test, prediction)
plt.xlabel('Y_test', fontsize=20)
plt.ylabel('prediction', fontsize=20)
plt.show()

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
file = open('reg_models.pkl', 'wb')
pickle.dump(models_list, file)
file.close()
