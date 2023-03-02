# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:58:08 2023

@author: Ahmad Al Musawi
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn import metrics
# from sklearn_extra.cluster import KMedoids
# from sklearn_extra.cluster import density_scan
# from sklearn_extra.cluster import hierarchical_consensus_clustering
# from sklearn_extra.cluster import k_medoids
# from sklearn_extra.cluster import pairwise_fast
# from sklearn_extra.cluster import spectral_clustering
# from sklearn_extra.cluster import spectral_embedding
# from sklearn_extra.cluster import vector_quantization

# import numpy as np

# from sklearn_extra.metrics import silhouette_score as sil


# Load the Iris dataset
iris = load_iris()
X = iris.data

real_labels = iris.target
k = len(set(real_labels))

# Create a KMeans clustering model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Compute the F1 score using the true labels and predicted labels
f1 = f1_score(real_labels, kmeans.labels_, average='weighted')
print("F1 score:", f1)


# Compute the adjusted Rand index using the true labels
ari = adjusted_rand_score(iris.target, kmeans.labels_)
print("Adjusted Rand index:", ari)

# Compute the silhouette score using the true labels and predicted labels
s1 = metrics.silhouette_score(X, real_labels, metric='euclidean')
s2 = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
print("Silhouette score (true labels):", s1)
print("Silhouette score (predicted labels):", s2)

# Compute the Calinski-Harabasz index
ch = metrics.calinski_harabasz_score(X, kmeans.labels_)
print("Calinski-Harabasz index:", ch)

# Compute the Davies-Bouldin index
db = metrics.davies_bouldin_score(X, kmeans.labels_)
print("Davies-Bouldin index:", db)


# Compute the normalized mutual information
nmi = metrics.normalized_mutual_info_score(iris.target, kmeans.labels_)
print("Normalized mutual information:", nmi)

