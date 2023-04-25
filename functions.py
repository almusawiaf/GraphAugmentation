# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:54:10 2023

@author: almusawiaf

Common functionalities...
"""
import networkx as nx
from sklearn import metrics
import itertools
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from information import *

def merge_tables(h):
    '''4/6/2023 merging the tables into one'''
    
    dfs = [pd.read_csv(f'data/{files[h]}/{ALL[h][i]}')     for i in [0,1,3]]
    total = dfs[0]
    for df in [dfs[1], dfs[2]]:
        features = list(total.keys())
        sub_features = list(df.keys())
        exclude = list(set(features).intersection(sub_features))
        new_features = [i for i in df if i not in exclude] + ['vha_id']
        total = pd.merge(total, df[new_features], on='vha_id')
    duplicates = total.columns[total.columns.duplicated()].tolist()

    # print the list of duplicate features
    print(duplicates)
    total.to_csv(f'data/{files[h]}/all.csv', index=False)
    

def spliting_the_table(df):
    common_features = ['vha_id', 'center_name', 'cancer_type']
    
    numerical_features = list(df.select_dtypes(include=['int', 'float']).columns)
    numerical_features = list(set(numerical_features + common_features))

    categorical_features = list(df.select_dtypes(include=['object']).columns)
    categorical_features = list(set(categorical_features + common_features))

    return df[categorical_features], df[numerical_features]

def remove_QM(h):
    df = pd.read_csv(f'data/{files[h]}/categorical.csv')
    qm = list(pd.read_csv(f'data/{files[h]}/QualityMeasures.csv')['QualityMeasures'])
    
    df[[i for i in qm + ['vha_id']]].to_csv(f'data/{files[h]}/QM data.csv')    
    df[[i for i in df if i not in qm ]].to_csv(f'data/{files[h]}/categorical data.csv')  
    
    
    

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

def plotting(df, f, l, d, v):
    ''' f: file name, 
        l: cluster id
        d: distance
        v: version (all vs one feature at a time)'''
    import matplotlib.pyplot as plt
    
    # define the list of rows to plot
    rows = ['Pass', 'Fail', 'Excluded']
    df = df.loc[rows]
    df = df.transpose()
    # filter the dataframe by the rows list and create the plot
    df.plot(kind='bar', stacked=True)
    
    # set the plot title and axis labels
    plt.title(f'Patient Outcome Rates for {f} SNF cluster {l}')
    plt.xlabel('Quality Measures')
    plt.ylabel('Rate (%)')
    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'data/imgs/{f}/{v} {d} cluster{l}.png', dpi = 300)

    # show the plot
    plt.show()
    
    
def comparing(h):
    '''
        h: 0 for lung, 1 for prostate
        df1: SNF based clusters of patients
        df2: SNF-QM outcome-based clusters of patients
        aim: identify common patients between each cluster...'''
    file = ['Lung','Prostate'][h]
    outcome = ['Pass', 'Fail', 'Excluded']
    Prostate_QMs = ['vha_id','QualityMeasure1','QualityMeasure10','QualityMeasure11','QualityMeasure12','QualityMeasure13','QualityMeasure14','QualityMeasure15','QualityMeasure15_color','QualityMeasure16','QualityMeasure17A','QualityMeasure17B','QualityMeasure18','QualityMeasure19','QualityMeasure2','QualityMeasure24','QualityMeasure3','QualityMeasure4','QualityMeasure5','QualityMeasure6','QualityMeasure7','QualityMeasure8','QualityMeasure9']
    Lung_QMs = ['vha_id','QualityMeasure1','QualityMeasure10','QualityMeasure11','QualityMeasure12','QualityMeasure13','QualityMeasure14','QualityMeasure15','QualityMeasure15Chemo','QualityMeasure15RT','QualityMeasure15Surgery','QualityMeasure16','QualityMeasure17','QualityMeasure18','QualityMeasure19','QualityMeasure19_color','QualityMeasure2','QualityMeasure20','QualityMeasure21A','QualityMeasure21B','QualityMeasure22','QualityMeasure23','QualityMeasure24','QualityMeasure27','QualityMeasure3','QualityMeasure4','QualityMeasure5','QualityMeasure6','QualityMeasure7','QualityMeasure8A','QualityMeasure8B','QualityMeasure9']
    QM = [Lung_QMs, Prostate_QMs][h][1:]
    
    df1 = pd.read_csv(f'data/Processed/One SNF for all tables/{file}/clusters.csv')
    total_pearson = {}
    total_spearman = {}
    for qm in QM:
        print('---------------------------------------------------------------')
        print(f'                 <  {qm}  >                            ')
        df2 = pd.read_csv(f'data/Processed/Separated_outcome_clustering/{file}/{qm}.csv')
        # selected_patients = df2.set_index('vha_id')[qm].to_dict()
        df_combined = pd.merge(df1[['vha_id',qm]], df2, on='vha_id')
        # print(df_combined)
        pearson_results = {}
        spearman_results = {}
        
        for i in set(df_combined[f'{qm}_y']):
            sub_df = df_combined[df_combined[f'{qm}_y'] == i]
            print(f'Outcome: {i}')
            cluster1 = list(sub_df[f'{qm}_x'].values)
            cluster2 = list(sub_df['Cluster'].values)
            if len(cluster1)>2 and len(cluster2)>2:
                pearson_coef, p_value = pearsonr(cluster1, cluster2)
                print(f'    Pearson  correlation = {pearson_coef}\tP value = {p_value}')
        
                spearman_coef, p_value = spearmanr(cluster1, cluster2)
                print(f'    Spearman correlation = {spearman_coef}\tP value = {p_value}')
                pearson_results[i] = pearson_coef
                spearman_results[i] = spearman_coef
            else:
                print('mushkila.....')
        total_pearson[qm] = pearson_results
        total_spearman[qm] = spearman_results
        print()
    pd.DataFrame(total_pearson).to_csv('data/Processed/Results of pearson correlation.csv', index=False)
    pd.DataFrame(total_spearman).to_csv('data/Processed/Results of spearman correlation.csv', index=False)
    return total_pearson

def PCA_model(X, y=None, n = 2):
#     print("PCA model")
    pca = PCA(n_components=n)
    pca.fit(X)
    X_pca = pca.transform(X)
    print(pca.explained_variance_ratio_)
    selected_features = pca.components_
#     print(f'PCA\tNoF = {len(selected_features)}')
    return X_pca

def Kernel_PCA(X, y=None, n = 2):
    from sklearn.decomposition import KernelPCA
#     print("Kernal PCA model")
    pca = KernelPCA(n_components=n, kernel='rbf')
    pca.fit_transform(X)
    return pca

def CE_Model(X, y=None, n=2):
    # Compute the spectral embedding using the Gaussian kernel
    sigma = 0.1
    embedding_gaussian = SpectralEmbedding(n_components=n, affinity='rbf', gamma=1 / (2 * sigma ** 2))
#     embedding = embedding_gaussian(n_components=n)
    X_CE = embedding_gaussian.fit_transform(X)
    return X_CE

def CE2(X, y=None, n=2):
    embedding = SpectralEmbedding(n_components=n, affinity='nearest_neighbors', n_neighbors=10, eigen_solver='arpack')
    X_CE = embedding.fit_transform(X)
#     print(f'CE2\tOld shape = {X.shape}\t\t new shape = {X_CE.shape}\t\t components = {n}')
    return X_CE

def LLE(X, y=None, n=2):
    from sklearn.manifold import LocallyLinearEmbedding
#     print('CE Model: Locally Linear Embedding')
    embedding = LocallyLinearEmbedding(n_components=n, n_neighbors=10)
    X_CE = embedding.fit_transform(X)
#     print(f'LLE\tOld shape = {X.shape}\t\t new shape = {X_CE.shape}\t\t components = {n}')
    return X_CE

def Isomap(X, y=None, n=2):
    from sklearn.manifold import Isomap
#     print('CE Model: Isomap')
    embedding =  Isomap(n_components=n, n_neighbors=10)
    X_CE = embedding.fit_transform(X)
#     print(f'ISOMAP\tOld shape = {X.shape}\t\t new shape = {X_CE.shape}\t\t components = {n}')
    return X_CE

def TSNE(X, y=None, n=2):
    from sklearn.manifold import TSNE
#     print('CE Model: TSNE')
    embedding = TSNE(n_components=2, perplexity=30, n_iter=1000)
    X_CE = embedding.fit_transform(X)
#     print(f'TSNE\tOld shape = {X.shape}\t\t new shape = {X_CE.shape}\t\t components = {n}')
    return X_CE



def CFS(X, y, n=2):
#     print('CFS Model')
    selector = SelectKBest(score_func=f_regression, k=n)
    X_new = selector.fit_transform(X, y)
    return X_new

def LLCFS(X, y=None,n=2):
#     print('LLCFS Model')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def ILFS(X, y, n=2):
    # create a linear regression model
#     print('ILFS Model')
    model = LinearRegression()
    
    # define the search space
    k_features = np.arange(1, X.shape[1]+1)
    
    # create a sequential feature selector object
    selector = SequentialFeatureSelector(model, k_features=k_features, forward=True, scoring='r2', cv=5)
    
    # perform incremental feature selection
    selector.fit(X, y)
    
    # print the selected feature indices
    print("Indices of selected features:", selector.k_feature_idx_)
    return selector.k_feature_idx_

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score