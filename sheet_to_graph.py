# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:43:20 2023

@author: almusawiaf

Convert a tabular data into a graph

"""
import networkx as nx
from itertools import combinations
from scipy.spatial import distance
import os

class DTNode:
    def __init__(self, name, data, ident):
        self.name = name
        self.data = data
        self.ident = ident
    
    def batch_processing(N):
        results = []
        for u,v in N:    
            vector1 = df[df['vha_id']==v].drop('vha_id', axis=1).to_numpy().tolist()[0]
            vector2 = df[df['vha_id']==u].drop('vha_id', axis=1).to_numpy().tolist()[0]
            d = distance.euclidean(vector1,vector2) 
            results.append([v, u, d])
        return results


G1 = nx.Graph()
G1.add_nodes_from(list(df['vha_id']))
Nodes = list(G1.nodes())
comb = list(combinations(Nodes, 2))

#     # if jaccard(list(vector1), list(vector2 ))!=0:
#     # if hamming(vector1, vector2)!=0:

def split_list_into_batches(lst, batch_size):
    batches = []
    for i in range(0, len(lst), batch_size):
        batch = lst[i:i+batch_size]
        batches.append(batch)
    return batches

my_list = list(combinations(Nodes, 2))

batch_size = 1000

batches = split_list_into_batches(my_list, batch_size)





results = []
import multiprocessing
if __name__ == '__main__':
    with multiprocessing.Pool(processes=8) as pool:
    # Divide the list of numbers into chunks and apply the sum_list function to each chunk in parallel
        results = pool.map(batch_processing, batches)

final = []
for i in results :
    for j in i:
        final.append(j)

G1 = nx.Graph()
for u, v, w in final:
    G1.add_edge(u, v, weight = w)

file_name = DB[2].split('.')[0]
nx.write_gml(G1, f'{os.getcwd()}\graphs\{file_name}.gml')
