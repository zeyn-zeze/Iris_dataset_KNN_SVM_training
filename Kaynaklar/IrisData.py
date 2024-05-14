#importing module 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


'''downlaod iris.csv from https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'''
#Load Iris.csv into a pandas dataFrame.
iris = pd.read_csv("iris2.csv")
print(iris.head(5))

#dropping the species column to make the inputs 
X = iris.drop(['species'],axis = 1)
#taikng species column to make labels
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

# printing the shape of testing and training data only for understanding means
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Now train the data using k nearest neighbour algorithms
scores = []
knn = KNeighborsClassifier(n_neighbors=12) #you can check for other n_neighbours, Try
knn.fit(X_train, y_train) #fitting the dataset in knn
y_pred = knn.predict(X_test)
scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)

#Checking is my model predicting right or not
print(knn.predict([[5.1,3.5,1.4,0.2]])) #setosa
print(knn.predict([[5.9,3.0,5.1,1.8]])) #versicolor
print(knn.predict([[3.0, 3.6, 1.3, 0.25]])) #Setosa


