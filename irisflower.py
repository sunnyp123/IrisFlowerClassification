# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 19:05:30 2018

@author: Sunny Parihar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()
#Parameters of iris flower can be expressed as dataframe.
iris_data = iris.data
iris_data = pd.DataFrame(iris_data,columns = iris.feature_names)
iris_data['class']= iris.target
iris_data.head()
iris.target_names
#Divide the data into training set and test set.
from sklearn.cross_validation import train_test_split
X = iris_data.iloc[:,[0,1,2,3]].values
Y = iris_data.iloc[:,4].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=42)
#Training the model with different classifications.
#1.)Logistic Regression.
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state=42)
classifier1.fit(X_train,Y_train)
Y_pred1 = classifier1.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test,Y_pred1))
cm1 = confusion_matrix(Y_test,Y_pred1)
#2.)K Nearest Neighbors.
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
classifier2.fit(X_train,Y_train)
Y_pred2 = classifier2.predict(X_test)
print(accuracy_score(Y_test,Y_pred2))
cm2 = confusion_matrix(Y_test,Y_pred2)
#3.) Support Vector Machine.
from sklearn.svm import SVC
classifier3 = SVC(kernel='linear',random_state=0)
classifier3.fit(X_train,Y_train)
Y_pred3 = classifier3.predict(X_test)
a = accuracy_score(Y_test,Y_pred3)
print(a)
cm3 = confusion_matrix(Y_test,Y_pred3)
#4.) Random Forest Classifier.
from sklearn.ensemble import RandomForestClassifier
classifier4 = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 42)
classifier4.fit(X_train,Y_train)
Y_pred4 = classifier4.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(Y_test,Y_pred4))
cm4 = confusion_matrix(Y_test,Y_pred4)
