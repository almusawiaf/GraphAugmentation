# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:49:20 2023

@author: Ahmad Al Musawi
"""

import pandas as pd
from sklearn.cluster import KMeans
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

# load the data
data = pd.read_csv('breast_cancer_data.csv')

# perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(data[['age', 'tumor_size', 'hormone_receptor_status']])

# add cluster assignments to the data
data['cluster'] = kmeans.labels_

# fit a Cox proportional hazards model to the survival data
cph = CoxPHFitter()
cph.fit(data[['time_to_event', 'event', 'cluster', 'age', 'tumor_size', 'hormone_receptor_status']], duration_col='time_to_event', event_col='event')

# calculate expected survival times for each patient in each cluster
data['expected_survival'] = cph.predict_expectation(data[['time_to_event', 'event', 'cluster', 'age', 'tumor_size', 'hormone_receptor_status']])

# plot the Kaplan-Meier survival curves for each cluster
kmf = KaplanMeierFitter()
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster']==cluster]
    kmf.fit(cluster_data['time_to_event'], cluster_data['event'], label='Cluster {}'.format(cluster))
    kmf.plot()

# perform the log-rank test to compare the survival distributions of the clusters
results = logrank_test(data['time_to_event'], data['cluster'], data['event'])
print(results.summary)
