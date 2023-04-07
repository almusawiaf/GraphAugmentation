# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:26:27 2023
@author: Ahmad Al Musawi

Patient Similarity Networks
using HINGE dataset
PRE-PROCESSING STAGE
"""

import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import jaccard
from sklearn.metrics import DistanceMetric
import math
from scipy.spatial import distance
import os
from snfpy.snf import snf
from information import *


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
    '''convert categorical features into ONE-HOT ENCODING... r, g, b r = 1,0,0 and so on'''
    cols = [c for c in df.columns if df[c].dtype.kind not in 'iufcb' and c!='vha_id']
    df[cols] = df[cols].fillna('aa')
    dummies = pd.get_dummies(df[cols])
    # concatenate the original dataframe with the dummies dataframe
    new_df = pd.concat([df.drop(cols, axis=1), dummies], axis=1)
    return new_df

def MinMax(df):
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    cols_to_normalize = [i for i in df if df[i].dtype.kind in 'iufcb']
    # Apply the scaler to the selected columns
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    # Print the normalized DataFrame
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

#----------------------------------------------------------------------------
if __name__=="__main__":
        
    #Handling each sub-dataset at a time...
    # our_path = 'E:/Research Projects/Complex Networks Researches/PSN Patient Similarity Network'
    h = int(input('enter 0 for lung.. 1 for prostate'))
    selected_DB = [DB_lung, DB_prostate][h]
    
    features = []
    for db in selected_DB:
        df = pd.read_csv(f'data/{db}')
        for i in df:
            features.append(i)
    
    QMs = [Lung_QMs, Prostate_QMs][h]
    
    
    for d in selected_DB:
        file_name = d.split('.')[0]
        print(f'Processing .... {file_name}')
    
        # df = pd.read_csv(f'data/{DB[d]}', skiprows=skipping[d])
        df = pd.read_csv(f'data/{d}')
        df = df.drop([ i for i in ['center_name', 'center_id'] + QMs if i in df], axis=1)
        # print(df.shape)
        
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
        
        df = MinMax(df)
        
        df.to_csv(f'data/processed/P3_{file_name}.csv', index=False)
    
    # # -------------------------------------------------------------------------
