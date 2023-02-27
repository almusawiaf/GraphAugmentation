# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:26:27 2023
@author: Ahmad Al Musawi

Patient Similarity Networks
using HINGE dataset
"""

import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import jaccard
from sklearn.metrics import DistanceMetric
import math
from scipy.spatial import distance
import os
from snfpy.snf import snf

def hamming(vector1, vector2):
    return sum(1 for v1, v2 in zip(vector1, vector2) if v1!=v2)

def remove_correlated(df):
    corr_matrix = df.corr().abs()    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    
    # Find features with correlation greater than 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    # Drop the highly correlated features
    df = df.drop(to_drop, axis=1)    
    return df

def dropping_cols(df, p=80):
    # count the number of NaN values in each column
    nan_counts = df.isna().sum()    
    # calculate the percentage of NaN values in each column
    nan_percentages = nan_counts / len(df) * 100 
    # get the list of columns to drop
    cols_to_drop = nan_percentages[nan_percentages > p].index.tolist()
    # drop the columns with more than 80% NaN values
    df = df.drop(cols_to_drop, axis=1)
    return df    

def preprocessing(df):
    '''convert categorical features into dummies... r, g, b r = 1,0,0 and so on'''
    cols = [c for c in df.columns if df[c].dtype.kind not in 'iufcb' and c!='vha_id']
    df[cols] = df[cols].fillna('aa')
    dummies = pd.get_dummies(df[cols])
    # concatenate the original dataframe with the dummies dataframe
    new_df = pd.concat([df.drop(cols, axis=1), dummies], axis=1)
    return new_df


def encoding(df):
    df = df.fillna('aa')

    cols = [c for c in df.columns if df[c].dtype.kind not in 'iufcb']
    
    #showing the uniques values for each column
    for col in cols :
        if col!='vha_id':
            
            if not df[col].dtype.kind in 'iufcb':

                # print(f"Non-numeric values in {col}: {df[~df[col].apply(lambda x: isinstance(x, (int, float)))][col]}")
                df[col] = df[col].astype(str)
                encoder = OrdinalEncoder(categories=[sorted(df[col].unique())])
                encoder.fit(df[[col]])
                # Transform the data
                df[[col]] = encoder.transform(df[[col]])
                # print(f'{col}:\t{unique_vals}')
    return df

def batch_processing(N, df):
    results = []
    for u,v in N:    
        vector1 = df[df['vha_id']==v].drop('vha_id', axis=1).to_numpy().tolist()[0]
        vector2 = df[df['vha_id']==u].drop('vha_id', axis=1).to_numpy().tolist()[0]
        d = distance.euclidean(vector1,vector2) 
        results.append([v, u, d])
    return results


def split_list_into_batches(lst, batch_size):
    batches = []
    for i in range(0, len(lst), batch_size):
        batch = lst[i:i+batch_size]
        batches.append(batch)
    return batches


#skipping used to skip some unuseful rows at each dataset.
# skipping = [2, 1, 0,0, 0,0,0, 0]

# d = 1
# cols = []
# dataset = []
# for d in range(len(skipping)):
#     data = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
#     dataset.append(data)
#     cols.append([i for i in data])

# # Showing number of shared patients
# shared_patient = []
# for df in dataset:
#     print(df['vha_id'])
#     shared_patient = shared_patient + list(df['vha_id'])

# for i in set(shared_patient):
#     print(i, shared_patient.count(i))

    

# #----------------------------------------------------------------------------
# #Shared and not shared features...
# shared = []
# for i in range(len(cols)-1):
#     for j in range(i+1, len(cols)):
#         shared.append([DB[i], DB[j], list(set(cols[i]) & set(cols[j]))])

# not_shared = []
# for i in range(len(cols)-1):
#     for j in range(i+1, len(cols)):
#         shared.append([item for item in cols[i] if item not in cols[j]])



#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#Handling each sub-dataset at a time...
our_path = 'E:/Research Projects/Complex Networks Researches/PSN Patient Similarity Network'
DB_lung = ["temp-lung-dvh.csv",
            "washu_measures_lung_all.csv",
            "washu_measures_lung_all_dvh_flag.csv",
            "washu_measures_lung_all_dvh_value.csv"]
DB_prostate = ["temp-prostate-dvh.csv",
                "washu_measures_prostate_all.csv",
                "washu_measures_prostate_all_dvh_flag.csv",
                "washu_measures_prostate_all_dvh_value.csv"]

selected_DB = DB_lung

features = []
for db in selected_DB:
    df = pd.read_csv(f'data/{db}')
    for i in df:
        features.append(i)



for d in selected_DB:
    file_name = d.split('.')[0]
    print(f'Processing .... {file_name}')

    # df = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
    df = pd.read_csv(f'data/{d}')
    df = df.drop([ 'center_name', 'center_id'], axis=1)
    print(df.shape)
    
    
    # drop the rows and cols with more than 80% NaN values
    df = dropping_cols(df)
    df = df.dropna(thresh=len(df.columns) * 0.2)
    df = df.drop_duplicates()
    
    # # Converting categorial into ordinal values
    # df = encoding(df)
    # print(df.shape)
    
    # Preprocessing
    print('before preprocessing', df.shape)
    df = preprocessing(df)
    print('after preprocessing', df.shape)
    
    # CORRELATION AND FEATURES REMOVAL
    df = remove_correlated(df)
    print('after removing correlated features', df.shape)
    
    df.to_csv(f'{our_path}/data/processed/P_{file_name}.csv', index=False)

# # -------------------------------------------------------------------------
# # My processing....

# Nodes = list(df['vha_id'])
# comb = list(combinations(Nodes, 2))


# my_list = list(combinations(Nodes, 2))

# batch_size = 1000

# batches = split_list_into_batches(my_list, batch_size)

# num_cores = multiprocessing.cpu_count()

# results = []
# if __name__ == '__main__':
#     with multiprocessing.Pool(processes=num_cores) as pool:
#     # Divide the list of numbers into chunks and apply the sum_list function to each chunk in parallel
#         results = pool.map(batch_processing, batches)

# final = []
# for i in results :
#     for j in i:
#         final.append(j)


# pd.DataFrame(final, columns=['source', 'target', 'weight']).to_csv(f'E:\Research Projects\Complex Networks Researches\PSN Patient Similarity Network\graphs\{file_name}.csv')


