#!/usr/bin/env python
# coding: utf-8

# In[1]:


from information import *
from preprocessing import *
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from snfpy.snf import snf
import snf

no_scale = ['vha_id', 'center_name', 'cancer_type']
def extract_features(df):
    return [i for i in df if i not in no_scale]
    


# In[2]:


def scaling(df):
    # Create an instance of StandardScaler
    features_to_scale = extract_features(df)
    scaler = StandardScaler()

    # Fit the scaler to the selected features
    scaler.fit(df[features_to_scale])

    # Transform the selected features using the scaler
    scaled_features = scaler.transform(df[features_to_scale])

    # Create a new dataframe with the scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

    # Add the unscaled features to the new dataframe
    df_scaled[no_scale] = df[no_scale]
    return df_scaled


# In[3]:


#handling numerical using standard scaler
df1 = pd.read_csv('data/Lung/numerical.csv', index_col=0)
df2 = pd.read_csv('data/Prostate/numerical.csv', index_col=0)

df1 = df1.fillna(0)
df2 = df2.fillna(0)

scaled_lung = scaling(df1)
scaled_prostate = scaling(df2)


# In[4]:


scaled_prostate


# In[5]:


#Dropping columns with missed values of more than 80%

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

print('scaled lung shape is ', scaled_lung.shape)
scaled_lung = dropping_cols(scaled_lung)
print('scaled lung shape is ', scaled_lung.shape)

print('scaled prostate shape is ', scaled_prostate.shape)
scaled_prostate = dropping_cols(scaled_prostate)

print('scaled prostate shape is ', scaled_prostate.shape)


# In[6]:


#dropping rows with missed values of more than 80%

print('scaled lung shape is ', scaled_lung.shape)
scaled_lung.dropna(thresh=len(scaled_lung.columns) * 0.2)
print('scaled lung shape is ', scaled_lung.shape)

print('scaled prostate shape is ', scaled_prostate.shape)
scaled_prostate.dropna(thresh=len(scaled_prostate.columns) * 0.2)
print('scaled prostate shape is ', scaled_prostate.shape)


# In[7]:


#dropping duplicate
scaled_lung = scaled_lung.drop_duplicates()
scaled_prostate = scaled_prostate.drop_duplicates()
print('scaled lung shape is ', scaled_lung.shape)
print('scaled prostate shape is ', scaled_prostate.shape)


# In[8]:


#removing correlated features with more than 0.9 correlation
scaled_lung = remove_correlated(scaled_lung)
scaled_prostate = remove_correlated(scaled_prostate)
print('scaled lung shape is ', scaled_lung.shape)
print('scaled prostate shape is ', scaled_prostate.shape)


# In[9]:


scaled_lung.head()


# In[10]:


#Working on Categorical dataframes
c_lung = pd.read_csv('data/Lung/categorical data.csv', index_col=0)
c_prostate = pd.read_csv('data/Prostate/categorical data.csv', index_col=0)
print(c_prostate.head())


# In[11]:


for f in c_lung:
    c_lung[f] = c_lung[f].fillna(c_lung[f].mode()[0])


# In[12]:


c_lung


# In[13]:


Nodes1 = list(c_lung['vha_id'])
scaled_lung = scaled_lung[extract_features(scaled_lung)]
scaled_prostate = scaled_prostate[extract_features(scaled_prostate)]

Lung_DB = [scaled_lung, c_lung]
Prostate_DB = [scaled_prostate]


# In[15]:



affinity_networks = snf.make_affinity(Lung_DB, metric=dist, K=20, mu=0.5)
fused_network = snf.snf(affinity_networks, K=20)
best, second = snf.get_n_clusters(fused_network)

labels = spectral_clustering(fused_network, n_clusters=best)
SNF = ff.make_graph(fused_network, selected_patients)


# In[ ]:




