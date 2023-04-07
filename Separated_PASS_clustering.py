# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:36:56 2023

@author: almusawiaf

Method 4:
    clustering based on Pass, Fail and Exclude seperatedly...

Objective:
    1- exclude the QM from the tables
    2- construct the PSGs for each table and merge them into one SNF
    3- for each QM, extract three subgraphs for those patient with QM = pass, fail and exclude
    4- for each of those subgraphs, 
        cluster them and 
        report patient, patient cluster, patient QM
    
"""

import pandas as pd
import numpy as np
import snf
from sklearn.cluster import spectral_clustering
import functions as ff
import networkx as nx
from information import *

    
my_datasets = []
    
if __name__=="__main__":

    h = int(input('0 - DB_lung\n1- DB_Prostate\n'))
    
    selected_DB = ALL[h]
    
    QualityMeasures = pd.read_csv(f'data/Data/{files[h]}/{selected_DB[1]}')[QM[h]]
    
    # version = ['P2','P3'][int(input('0: P2 files, 1: P3 files'))]
    version = 'P3' #Implementation on P3 processed only

    for d in range(len(selected_DB)):
        file_name = selected_DB[d].split('.')[0]
        print(f'Processing .... {file_name}')
    
        # df = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
        df = pd.read_csv(f'data/processed/{version}_{selected_DB[d]}')
        df = df.fillna(0)
    
        my_datasets.append(df)
    
        
    dist = distance[0]
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

    for qm in QM[h][1:]:
        # current_QM = QualityMeasures.loc[:, ['vha_id',qm]]
        current_QM = QualityMeasures[['vha_id',qm]].fillna('Excluded')
        
        print(f'current_QM : {qm}\n')
        # -----------------------------------------------------------------------------
        # Outcome = {'pass': [patient1, patient2...etc], 'Fail':....}
        Outcome = {}
        for outcome in set(current_QM[qm]):
            dfi = current_QM.loc[current_QM[qm]==outcome, ['vha_id']]
            dfi = dfi['vha_id'].tolist()
            Outcome[outcome] = dfi
        print(f'outcome = {set(current_QM[qm])}\n')
        # -----------------------------------------------------------------------------
        print('Network Fusion------------------------------------------\n')
        affinity_networks = snf.make_affinity(new_DB, metric=dist, K=20, mu=0.5)
        fused_network = snf.snf(affinity_networks, K=20)
        SNF = ff.make_graph(fused_network, selected_patients)
        # generating the different graphs for each outcome using the specific set of patients of that outcome.
        subSNF = {i: SNF.subgraph(Outcome[i]) for i in set(current_QM[qm])}
        best_subSNF = {}
        for i in set(current_QM[qm]):
            if len(subSNF[i].nodes())>5:
                best, _ = snf.get_n_clusters(np.asarray(nx.to_numpy_matrix(subSNF[i])))
            else:
                best = 1
            best_subSNF[i] = best
        
        # labels_subSNF = {i: spectral_clustering(np.asarray(nx.to_numpy_matrix(subSNF[i])), n_clusters=best_subSNF[i]) for i in set(current_QM[qm])}
        labels_subSNF  = {}
        for i in set(current_QM[qm]):
            temp = np.asarray(nx.to_numpy_matrix(subSNF[i]))
            labels_subSNF[i] = spectral_clustering(temp, n_clusters=best_subSNF[i])
        # assign cluster id to the patients
        final_df = pd.DataFrame(columns=['vha_id', qm, 'Cluster'])
        for i in set(current_QM[qm]):
            V = list(subSNF[i].nodes())
            C = labels_subSNF[i]
            for j in range(len(V)):
                # final_df = final_df.append({'vha_id': V[j], qm: i, 'Cluster': C[j]}, ignore_index=True)
                new_row = {'vha_id': V[j], qm: i, 'Cluster': C[j]}
                final_df = pd.concat([final_df, pd.DataFrame([new_row])], ignore_index=True)

        final_df.to_csv(f'data/processed/Separated_outcome_clustering/{files[h]}/{qm}.csv')
            
        
