# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:56:51 2023
URL : https://github.com/rmarkello/snfpy
@author: Ahmad Al Musawi
"""

from snf import datasets
import pandas as pd
import numpy as np
import snf
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms.community import k_clique_communities
import community
from sklearn import metrics
import itertools


def make_graph(sim, N, threshold=0):
    '''This method receive a similarity matrix and return a graph...'''
    # Create an empty undirected graph
    G = nx.Graph()
    
    # Add nodes to the graph
    n_nodes = sim.shape[0]
    G.add_nodes_from(N)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Add an edge between nodes i and j if similarity is above threshold
            if sim[i,j] > threshold:
                G.add_edge(N[i], N[j], weight=sim[i,j])
    return G


def K_Clique(g):
    # find all maximal cliques of size 3 or greater
    cliques = list(nx.find_cliques(g))
    
    # find all k-cliques of size k=3
    k_cliques = list(nx.k_clique_communities(g, 3))
    
    print("Maximal Cliques: ", cliques)
    print("3-Cliques: ", k_cliques)


def evaluation(Merged, PSGs):
    from sklearn.metrics import f1_score
    from sklearn.metrics import jaccard_score
    
    f1 = [f1_score(Merged, l) for l in PSGs]
    print(f'F1 score = {f1}')
    
    ja = [jaccard_score(Merged, l) for l in PSGs]
    print(f'Jaccard score = {ja}')


def evaluation2(data):
    comb = itertools.combinations(['SNF','PSG1','PSG2','PSG3','PSG4'], 2)
    silhouette = {}
    for i,j  in comb:
        silhouette[(i, j)] = metrics.silhouette_score(data[[i]], data[[j]],  metric = 'euclidean')
    print({i: set(data[i]) for i in ['SNF','PSG1','PSG2','PSG3','PSG4']})
    return silhouette

def Pvalue(data):
    from scipy import stats
    comb = itertools.combinations(['SNF','PSG1','PSG2','PSG3','PSG4'], 2)
    results = {}
    for i,j  in comb:
        results [(i, j)] = stats.ttest_ind(data[[i]], data[[j]])
    return results



DB_lung = ["temp-lung-dvh.csv",
            "washu_measures_lung_all.csv",
            "washu_measures_lung_all_dvh_flag.csv",
            "washu_measures_lung_all_dvh_value.csv"]


DB_prostate = ["temp-prostate-dvh.csv",
                "washu_measures_prostate_all.csv",
                "washu_measures_prostate_all_dvh_flag.csv",
                "washu_measures_prostate_all_dvh_value.csv"]


our_path = 'E:/Research Projects/Complex Networks Researches/PSN Patient Similarity Network'

ALL = [DB_lung, DB_prostate]
my_datasets = []

h = int(input('0 - DB_lung\n1- DB_Prostate\n'))
selected_DB = ALL[h]

for d in range(len(selected_DB)):
    file_name = selected_DB[d].split('.')[0]
    print(f'Processing .... {file_name}')

    # df = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
    df = pd.read_csv(f'data/processed/P_{selected_DB[d]}')
    df = df.fillna(0)

    my_datasets.append(df)


# -----------------------------------------------------------------------------
#                              Data Pre-processing

Nodes = [list(my_datasets[i]['vha_id']) for i in range(len(selected_DB))]
selected_patients= sorted(list(set(Nodes[0]).intersection(*Nodes[1:])))
new_DB = [df[df['vha_id'].isin(selected_patients)] for df in my_datasets]
new_DB = [df.fillna(0) for df in new_DB]
new_DB = [df.drop('vha_id', axis=1) for df in new_DB]

# -----------------------------------------------------------------------------
#                               Selecting the networks
chosen_DB = [0,3]
new_DB = [new_DB[i].values for i in chosen_DB]
# -----------------------------------------------------------------------------
#                               Network Fusion

affinity_networks = snf.make_affinity(new_DB, metric='euclidean', K=20, mu=0.5)
fused_network = snf.snf(affinity_networks, K=20)
best, second = snf.get_n_clusters(fused_network)

k = best
labels = spectral_clustering(fused_network, n_clusters=k)
labels_dict = {selected_patients[i]: labels[i] for i in range(len(selected_patients))}
# -----------------------------------------------------------------------------
#                              Networks Spectral clustering

similarity_DB = [cosine_similarity(d) for d in new_DB]
best_DB = [snf.get_n_clusters(d) for d in similarity_DB]
labels2= [spectral_clustering(similarity_DB[i], n_clusters=best_DB[i][0]) for i in range(len(best_DB)) ]

print([metrics.silhouette_score(similarity_DB[j], labels2[j],  metric = 'euclidean') for j in range(len(new_DB))])
print(metrics.silhouette_score(fused_network, labels, metric = 'euclidean'))



Gs = [make_graph(db, selected_patients) for db in similarity_DB]
# for i in range(len(similarity_DB)):
#     g = Gs[i]
#     pd.DataFrame(list(g.edges()), columns=['Source', 'Target']).to_csv(f'data/processed/Network {selected_DB[i]}', index=False)
# # -----------------------------------------------------------------------------

# # evaluation(labels, labels2)
# # # -----------------------------------------------------------------------------

# exp1 = pd.DataFrame(list(zip(selected_patients, labels, labels2[0],labels2[1],labels2[2],labels2[3])), columns=['Patients','SNF','PSG1','PSG2','PSG3','PSG4'])
# print('Original spectral clustering silhouette score:')
# print(evaluation2(exp1))
# print(sum([metrics.silhouette_score(exp1[['SNF']], exp1[[j]],  metric = 'euclidean') for j in ['PSG1','PSG2','PSG3','PSG4']])/4)
# -----------------------------------------------------------------------------

SNF = make_graph(fused_network, selected_patients)
Partitions = []
M = []
for g in [SNF]+Gs:
    partition = community.best_partition(g)
    Partitions.append(partition)
    modularity = community.modularity(partition, g)
    M.append(modularity)
print(f'Modularity = {M}')

clusters0 = {patient:[p[patient] for p in Partitions] for patient in selected_patients if patient not in ['541-SCLC-09','548-NSCLC-07']}
clusters1 = pd.DataFrame(clusters0).transpose()
clusters1.columns=['SNF','PSG1','PSG2','PSG3','PSG4']

print(metrics.silhouette_score(similarity_DB[0], clusters1['PSG1'], metric = 'euclidean'))
print(metrics.silhouette_score(similarity_DB[1], clusters1['PSG2'], metric = 'euclidean'))
print(metrics.silhouette_score(similarity_DB[2], clusters1['PSG3'], metric = 'euclidean'))
print(metrics.silhouette_score(similarity_DB[3], clusters1['PSG4'], metric = 'euclidean'))
print(metrics.silhouette_score(fused_network, clusters1['SNF'], metric = 'euclidean'))

# # evaluation(clusters1['SNF'].values, [clusters1['SNF'].values])
# # -----------------------------------------------------------------------------
# print('Community based partitioning (clustering) silhouette score:')
# print(evaluation2(clusters1))
    

