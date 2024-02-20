# Import The Dependencies
# Import Libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sbn
import sys
import random as rnd 

# Machine Learning Libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# Loading the data from csv file to a pandas DataFrame

customer_data = pd.read_csv(r"C:\Users\Dell\Desktop\AI\Machine Learning\Mall Customer ML clustering\Mall_Customers.csv")

print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# First five rows in a dataset

print(customer_data.head())
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Finding the number of rows and columns

print(customer_data.shape)
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Getting some information about the dataset
print(customer_data.info())
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Checking the missing values
print(customer_data.isnull().sum())
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Choosing the Annual Income & Spending Score column
X = customer_data.iloc[:,[3,4]].values
print(X)
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Choosing the number of clusters
# WCSS - Within Cluster sum of Squares
# Finding WCSS value for the different number of clusters
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Plot an Elbow Graph

sbn.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Optimum number of clusters = 5
# Training the k-Means Clustering Model
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=0)

# Return a lebel for each data point based on thier cluster
Y = kmeans.fit_predict(X)

print(Y)
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")

# Visualizing all the Clusters
# Plotting the Clusters

plt.figure(figsize=(10,10))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label= 'Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='blue', label= 'Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='yellow', label= 'Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label= 'Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='red', label= 'Cluster 5')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()