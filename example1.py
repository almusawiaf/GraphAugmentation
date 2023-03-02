# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:41:47 2023

@author: Ahmad Al Musawi
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn import metrics


# Load the Iris dataset
iris = load_iris()
X = iris.data

# Create a KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Compute the adjusted Rand index using the true labels
ari = adjusted_rand_score(iris.target, kmeans.labels_)

print("Adjusted Rand index:", ari)

targets = iris.target
labels = kmeans.labels_
print("F1 score:", f1_score(targets, labels, average='weighted'))

print('Silhouette score: ', metrics.silhouette_score(X, targets,  metric = 'euclidean') )
print('Silhouette score: ', metrics.silhouette_score(X, labels,  metric = 'euclidean') )
