# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:56:51 2023
URL : https://github.com/rmarkello/snfpy
@author: Ahmad Al Musawi
"""

import pandas as pd
import snf
from sklearn.cluster import spectral_clustering
from preprocessing import preprocessing as pp
from information import *

    
def rates(df, qm):
    # group by cluster and calculate the percentage of Fail and Pass for each cluster
    result = df.groupby('cluster')[qm].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    return result
    # # filter out the rows that have 'Fail' in the 'QualityMeasure11' column
    # result = result[result[qm] != 'Fail']
    
    # # pivot the result DataFrame to have the percentage of Pass and Fail for each cluster as columns
    # result = result.pivot(index='cluster', columns=qm, values='percentage').reset_index()
    
    # # rename the columns
    # result.columns.name = None
    # result = result.rename(columns={'Pass': 'pass_percentage', 'Fail': 'fail_percentage'})
    
    # # print the result
    # print(result)

def rates2(df,qm):
    counts = pd.crosstab(index=df['cluster'], columns=df[qm])

    # compute the percentage of each value count for each cluster
    percentage = counts.div(counts.sum(axis=1), axis=0).multiply(100).reset_index()
    
    # rename the columns and display the result
    percentage.columns.name = ''
    percentage.rename(columns={'cluster': f'{qm}\tCluster'}, inplace=True)
    return percentage

def iterating(f):
    import os
    import matplotlib.pyplot as plt
    folder_path = f'data/Processed/{f}'
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        if os.path.isfile(file_path):
            print(filename)
            ff = filename.split('.')[0]
            ff = ff.split('_')
            qm, clusters = ff[0], ff[1]
            df = pd.read_csv(file_path, index_col=0, usecols=lambda col: col != 'Unnamed')
            
            # set the 'Cluster' column as the index
            df.set_index('Cluster', inplace=True)
            
            # create a horizontal bar chart
            ax = df.plot(kind='bar', stacked=True)
            
            # set the title and axis labels
            ax.set_title(f'Patient Outcome Rates for {qm}')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Percentage')

            plt.tight_layout()
            plt.savefig(f'{folder_path}/imgs/{qm}.png', dpi = 300)
            
            # show the plot
            plt.show()
            # plotting(, qm, clusters, '','')


my_datasets = []
    
if __name__=="__main__":

    h = int(input('0 - DB_lung\n1- DB_Prostate\n'))
    
    selected_DB = ALL[h]
    
    QualityMeasures = pd.read_csv(f'data/{selected_DB[1]}')[QM[h]]
    
    # version = ['P2','P3'][int(input('0: P2 files, 1: P3 files...?'))]
    version = 'P3' #Implementation on P3 processed only

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
    #                       doing QM1: <QualityMeasure1>
    
    qm = QM[h][1] # QualityMeasures[0]
    current_QM = QualityMeasures.loc[:, ['vha_id',qm]]
    print(f'current_QM : {current_QM}')
    final_df = pd.DataFrame({'vha_id': selected_patients})
    
    for qm in QM[h][1:]:
        total_DB = new_DB
        # do not append QM to data and cluster...
        # current_QM = QualityMeasures.loc[:, ['vha_id',qm]]
        # ppQM = pp(QualityMeasures[qm].to_frame()).values
        # total_DB.append(ppQM)
        print('Network Fusion------------------------------------------')
        affinity_networks = snf.make_affinity(total_DB , metric=dist, K=20, mu=0.5)
        fused_network = snf.snf(affinity_networks, K=20)
        best, _ = snf.get_n_clusters(fused_network)
        
        labels = spectral_clustering(fused_network, n_clusters=best)
        print(qm, best)
        labels_dict = {'vha_id': selected_patients, qm: labels}
        labels_df = pd.DataFrame(labels_dict)
        final_df = pd.merge(final_df, labels_df, on='vha_id')
    final_df.to_csv(f'data/Processed/One SNF for all tables/{files[h]}/clusters.csv', index=False)
            # -----------------------------------------------------------------------------
            # results_df = pd.merge(labels_df, current_QM, on='vha_id')
            # rates2(results_df, qm).to_csv(f'data/Processed/{files[h]}/{qm}_{best}.csv')
