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
from preprocessing import preprocessing as pp
from information import *
import functions as ff



my_datasets = []
    
if __name__=="__main__":

    h = int(input('0 - DB_lung\n1- DB_Prostate\n'))
    kp =    input('Community Partitioning (y,n)?')
    
    selected_DB = ALL[h]
    
    QualityMeasures = pd.read_csv(f'data/{files[h]}/{selected_DB[1]}')[QM[h]]
    
    version = ['P2','P3'][int(input('0: P2 files, 1: P3 files'))]

    for d in range(len(selected_DB)):
        file_name = selected_DB[d].split('.')[0]
        print(f'Processing .... {file_name}')
    
        # df = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
        df = pd.read_csv(f'data/processed/{version}_{selected_DB[d]}')
        df = df.fillna(0)
    
        my_datasets.append(df)
    
        
    # -----------------------------------------------------------------------------
    dist = distance[0]
    # print('select distance:')
    # for i in range(len(distance)):
    #     print(f'{i} .. {distance[i]}')
    # dist = distance[int(input('Enter index ...'))]
    # -----------------------------------------------------------------------------
    #                              Data Pre-processing
    # -----------------------------------------------------------------------------
    print('Data pre-processing------------------------------------------')
    Nodes = [list(my_datasets[i]['vha_id']) for i in range(len(selected_DB))]
    
    selected_patients= sorted(list(set(Nodes[0]).intersection(*Nodes[1:])))
    QualityMeasures = QualityMeasures[QualityMeasures['vha_id'].isin(selected_patients)]
    
    new_DB = [df[df['vha_id'].isin(selected_patients)] for df in my_datasets]
    new_DB = [df.fillna(0) for df in new_DB]
    
    #dropping 
    new_DB = [df.drop('vha_id', axis=1) for df in new_DB]
    new_DB = [df.drop([i for i in QualityMeasures if i in df], axis=1) for df in new_DB]
    # -----------------------------------------------------------------------------
    #                               Selecting the networks
    new_DB = [new_DB[i].values for i in range(len(new_DB))]
    # -----------------------------------------------------------------------------
    #                       converting quality measures into dfs
    new_QM = [pd.DataFrame(QualityMeasures[i]) for i in QualityMeasures if i!='vha_id']
    new_QM = [pp(df).values for df in new_QM]
    _ = input('Quit...')
    # -----------------------------------------------------------------------------
    #                               Network Fusion
    # -----------------------------------------------------------------------------
    print('Network Fusion------------------------------------------')
    affinity_networks = snf.make_affinity(new_DB, metric=dist, K=20, mu=0.5)
    fused_network = snf.snf(affinity_networks, K=20)
    best, second = snf.get_n_clusters(fused_network)
    
    labels = spectral_clustering(fused_network, n_clusters=best)
    SNF = ff.make_graph(fused_network, selected_patients)
    labels_dict = {selected_patients[i]: labels[i] for i in range(len(selected_patients))}
    # -----------------------------------------------------------------------------
    #                              Networks Spectral clustering
    # -----------------------------------------------------------------------------
    print('Networks Spectral clustering------------------------------------------')
    similarity_DB = [cosine_similarity(d) for d in new_DB]
    best_DB = [snf.get_n_clusters(d) for d in similarity_DB]
    labels2= [spectral_clustering(similarity_DB[i], n_clusters=best_DB[i][0]) for i in range(len(best_DB)) ]
    
    ss1 = [metrics.silhouette_score(similarity_DB[j], labels2[j],  metric = 'euclidean') for j in range(len(new_DB))]
    ss2 = metrics.silhouette_score(fused_network, labels, metric = 'euclidean')
    print(f'Number of clusters in SNF = {best}')
    print(f'silhouette score = {ss1}')
    print(f'silhouette score = {ss2}')
    
    Gs = [ff.make_graph(db, selected_patients) for db in similarity_DB]
    
    # -----------------------------------------------------------------------------
    #                               Community Partitioning
    # -----------------------------------------------------------------------------
    if kp =='y':
        print('Community Partitioning------------------------------------------')
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
        # -----------------------------------------------------------------------------
        #                              Silhouette score
        # -----------------------------------------------------------------------------
        print(metrics.silhouette_score(similarity_DB[0], clusters1['PSG1'], metric = 'euclidean'))
        print(metrics.silhouette_score(similarity_DB[1], clusters1['PSG2'], metric = 'euclidean'))
        print(metrics.silhouette_score(similarity_DB[2], clusters1['PSG3'], metric = 'euclidean'))
        print(metrics.silhouette_score(similarity_DB[3], clusters1['PSG4'], metric = 'euclidean'))
        print(metrics.silhouette_score(fused_network, clusters1['SNF'], metric = 'euclidean'))
    # -----------------------------------------------------------------------------
        
    for l in range(best):
        p = [i for i in labels_dict if labels_dict[i]==l]
        dd = QualityMeasures.loc[QualityMeasures['vha_id'].isin(p)]
        results = {}
        for column in dd.columns[1:]:
            rates = dd.groupby(column)['vha_id'].count() / len(dd) * 100
            results[column] = rates.to_dict()
        df = pd.DataFrame(results)
        ff.plotting(df, files[h], l, dist, version)
        #saving the df...no need to save...
        # df.to_csv(f'data/processed/FailPassRates_{files[h]}_clusters{l}.csv')  
        
        
    
    # -----------------------------------------------------------------------------
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
    
    # # evaluation(clusters1['SNF'].values, [clusters1['SNF'].values])
    # # -----------------------------------------------------------------------------
    # print('Community based partitioning (clustering) silhouette score:')
    # print(evaluation2(clusters1))
