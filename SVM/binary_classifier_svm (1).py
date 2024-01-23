# -*- coding: utf-8 -*-
"""Binary_classifier_SVM.ipynb
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel = 'linear', C = 1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test, y_pred)
print(f"accuracy: {accuracy:.2f}")
print("classification_report:\n", report)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

X_visualization = X[:,:2]

svm_model_visualization = SVC(kernel = 'linear', C = 1.0)
svm_model_visualization.fit(X_visualization,y)

plot_decision_regions(X_visualization,y , clf=svm_model_visualization,legend = 2)
plt.xlabel('Sepal length')
plt.ylabel('sepal width')
plt.title('SVM decision regions for setosa classification')
plt.show()

