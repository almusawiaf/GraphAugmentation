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

import multiprocessing
import pandas as pd
from itertools import combinations
from scipy.spatial import distance


    
DB = ["temp-lung-dvh.csv",
    "temp-prostate-dvh.csv",
    "washu_measures_lung_all.csv",
    "washu_measures_lung_all_dvh_flag.csv",
    "washu_measures_lung_all_dvh_value.csv",
    "washu_measures_prostate_all.csv",
    "washu_measures_prostate_all_dvh_flag.csv",
    "washu_measures_prostate_all_dvh_value.csv"]

def remove_correlated(df):
    corr_matrix = df.corr().abs()    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))    
    # Find features with correlation greater than 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
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
skipping = [2, 1, 0,0, 0,0,0, 0]

d = 0
name = DB[d]

print(f'\ncurrent file is: {name.upper()}\n')
df = pd.read_csv(f'data/{name}', skiprows=skipping[d])
df = df.drop([i for i in  ['center_name', 'center_id'] if i in df], axis=1)
# print(df.shape)


# drop the rows and cols with more than 80% NaN values
df = dropping_cols(df)
df = df.dropna(thresh=len(df.columns) * 0.2)
df = df.drop_duplicates()

# Converting categorial into ordinal values
df = encoding(df)
# print(df.shape)

# CORRELATION AND FEATURES REMOVAL
df = remove_correlated(df)
print(df.shape)

_ = input('press any key...')
# -------------------------------------------------------------------------
Nodes = list(df['vha_id'])

my_list = list(combinations(Nodes, 2))

batch_size = 1000

batches = split_list_into_batches(my_list, batch_size)    

num_cores = multiprocessing.cpu_count()

results = []
if __name__ == '__main__':
    with multiprocessing.Pool(processes=num_cores) as pool:
    # Divide the list of numbers into chunks and apply the sum_list function to each chunk in parallel
        results = pool.map(batch_processing, batches)

final = []
for i in results :
    for j in i:
        final.append(j)


file_name = name.split('.')[0]

pd.DataFrame(final, columns=['saurce', 'target', 'weight']).to_csv(f'E:\Research Projects\Complex Networks Researches\PSN Patient Similarity Network\graphs\{file_name}.csv')
