import pickle
from Preprocessing import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Load data
df = read_data('AppleStore_training_classification.csv')

#dealing with missing values
df.dropna(axis=0, how='any', inplace=True)

#extract features and target
data, X, Y = extract_XY(df, 'rate')

#print(list(X)[-2])
#print(list(X)[6])

X = xc_preprocess(X, 6)
Y = yc_prerocesss(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

models_list = []
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#Logistic Regression
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# start = time()
# logreg = LogisticRegression(C=0.000001, penalty='l2')
# logreg.fit(X_train, y_train)
# print("Logistic reg (C=0.000001) training took %f seconds" % (time() - start))
# start = time()
# prediction = logreg.score(X_test, y_test)
# print("Logistic reg (C=0.000001) testing took %f seconds" % (time() - start))
# print('logistic reg (C=0.000001) accuracy: ' + str(prediction))

print("#")
start = time()
logreg = LogisticRegression(C=0.1, penalty='l2')
logreg.fit(X_train, y_train)
models_list.append(logreg)
print("Logistic reg (C=0.1) training took %f seconds" % (time() - start))
start = time()
prediction = logreg.score(X_test, y_test)
print("Logistic reg (C=0.1) testing took %f seconds" % (time() - start))
print('logistic reg (C=0.1) accuracy: ' + str(prediction))

# print("#")
# start = time()
# logreg = LogisticRegression(C=10, penalty='l2')
# logreg.fit(X_train, y_train)
# print("Logistic reg (C=10) training took %f seconds" % (time() - start))
# start = time()
# prediction = logreg.score(X_test, y_test)
# print("Logistic reg (C=10) testing took %f seconds" % (time() - start))
# print('logistic reg (C=10) accuracy: ' + str(prediction))

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#SVM
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

# start = time()
# clf = svm.SVC(C=0.000001)
# clf.fit(X_train, y_train)
# print("SVM (C=0.000001) training took %f second" % (time() - start))
# start = time()
# predection = clf.score(X_test, y_test)
# print("SVM (C=0.000001) testing took %f second" % (time() - start))
# print('SVM (C=0.000001) test accuracy: ' + str(predection))
# print('SVM (C=0.000001) train accuracy: ' + str(clf.score(X_train, y_train)))

print("#")
start = time()
clf = svm.SVC(C=0.1)
clf.fit(X_train, y_train)
models_list.append(clf)
print("SVM (C=0.1) training took %f second" % (time() - start))
start = time()
predection = clf.score(X_test, y_test)
print("SVM (C=0.1) testing took %f second" % (time() - start))
print('SVM (C=0.1) test accuracy: ' + str(predection))
#print('SVM (C=0.1) train accuracy: ' + str(clf.score(X_train, y_train)))

# print("#")
# start = time()
# clf = svm.SVC(C=1)
# clf.fit(X_train, y_train)
# print("SVM (C=1) training took %f second" % (time() - start))
# start = time()
# predection = clf.score(X_test, y_test)
# print("SVM (C=1) testing took %f second" % (time() - start))
# print('SVM (C=1) test accuracy: ' + str(predection))
# print('SVM (C=1) train accuracy: ' + str(clf.score(X_train, y_train)))

# print("#")
# start = time()
# clf = svm.SVC(C=100)
# clf.fit(X_train, y_train)
# print("SVM (C=100) training took %f second" % (time() - start))
# start = time()
# predection = clf.score(X_test, y_test)
# print("SVM (C=100) testing took %f second" % (time() - start))
# print('SVM (C=100) test accuracy: ' + str(predection))
# print('SVM (C=100) train accuracy: ' + str(clf.score(X_train, y_train)))

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#KNN
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

start = time()

error = []
acc = []
knn_models = []
#pred_i=0
for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_models.append(knn)
    # pred_i = knn.score(X_test, y_test)
    # error.append(pred_i)

print("KNN training took %f second" % (time() - start))
start = time()

for i in range(1, 50):
    pred_i = knn_models[i-1].predict(X_test)
    error.append(np.mean(pred_i != y_test))
    acc.append(np.mean(pred_i == y_test))

print("KNN testing took %f second" % (time() - start))

index = error.index(np.min(error))
print('KNN accuracy: ' + str(acc[index]))
models_list.append(knn_models[index])

plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#Decision Tree Classifier
print("#")
print("#")
print("#")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


# start = time()
# tree_clf = tree.DecisionTreeClassifier(max_depth=1)
# tree_clf.fit(X_train, y_train)
# print("Decision Tree (max_depth=1) training took %f second" % (time() - start))
# start = time()
# accuracy = tree_clf.score(X_test, y_test)
# print("Decision Tree (max_depth=1) testing took %f second" % (time() - start))
# print("Decision Tree (max_depth=1) accuracy: " + str(accuracy))

print("#")
start = time()
tree_clf = tree.DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)
print("Decision Tree (max_depth=3) training took %f second" % (time() - start))
models_list.append(tree_clf)
start = time()
accuracy = tree_clf.score(X_test, y_test)
print("Decision Tree (max_depth=3) testing took %f second" % (time() - start))
print("Decision Tree (max_depth=3) accuracy: " + str(accuracy))

# print("#")
# start = time()
# tree_clf = tree.DecisionTreeClassifier(max_depth=100)
# tree_clf.fit(X_train, y_train)
# print("Decision Tree (max_depth=100) training took %f second" % (time() - start))
# start = time()
# accuracy = tree_clf.score(X_test, y_test)
# print("Decision Tree (max_depth=100) testing took %f second" % (time() - start))
# print("Decision Tree (max_depth=100) accuracy: " + str(accuracy))

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

file = open('class_models.pkl', 'wb')
pickle.dump(models_list, file)
file.close()
