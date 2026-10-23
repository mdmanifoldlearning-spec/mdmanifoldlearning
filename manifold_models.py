# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:03:22 2021

@author: Shaodong
"""

from sklearn.manifold import SpectralEmbedding, Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
# import umap
import numpy as np
import pickle
import sys, os
import time
from utils import load_pickle, dump_pickle

"""
get UMAP embeddings
args:
    X (numpy array): 2D numpy array of precomputed distance matrix 
            (|visits| * |visits|).
    k (int): dimension that reduce to
    n_neighbors (int): a parameter in the network
    loc (str): location to store the embeddings
    used_method (str): used distance method. It will be used to store the 
            embeddings.
"""
def UMAP_embedding(X,k,n_neighbors,min_dist,loc=None,used_method = 'IC1CS2SS7',recalc=True):
    path = loc+used_method+'UMAP_k'+str(k)+'neighbor'+str(n_neighbors)+\
                'mindist'+str(min_dist)+'.pickle'
    if os.path.exists(path):
        recalc = False
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    X_embedded = umap.UMAP(n_neighbors = n_neighbors,min_dist = min_dist,\
                           n_components = k, metric= 'precomputed'\
                           ).fit_transform(X)
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded
"""
get tSNE embeddings
args:
    X (numpy array): 2D numpy array of precomputed distance matrix 
            (|visits| * |visits|).
    k (int): dimension that reduce to
    loc (str): location to store the embeddings
    used_method (str): used distance method. It will be used to store the 
            embeddings.
"""
def TSNE_embedding(X,k,perplexity,loc=None,used_method = 'IC1CS2SS7',recalc = True):
    path = loc+used_method+'TSNE_k'+str(k)+'perplexity'+str(perplexity)+'.pickle'
    if os.path.exists(path):
        recalc = False
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    X_embedded = TSNE(n_components = k, perplexity=perplexity,\
                      metric='precomputed',method='exact').fit_transform(X)
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded
"""
get Laplacian Eigenmap embeddings
args:
    X (numpy array): 2D numpy array of precomputed distance matrix 
            (|visits| * |visits|).
    k (int): dimension that reduce to
    sigma (float): a parameter in affine matrix
    loc (str): location to store the embeddings
    used_method (str): used distance method. It will be used to store the 
            embeddings.
"""
def Eigenmap_embedding(X, k, n_neighbors, sigma = 0.7, loc=None,\
                       used_method = 'IC1CS2SS7',recalc = True,full_connect=False):
    if full_connect == False:
        path = loc+used_method+'Eigenmap_k'+str(k)+\
                    'neighbor'+str(n_neighbors)+'.pickle'
    else:
        path = loc+used_method+'Eigenmap_k'+str(k)+'sig'+str(sigma)+'fullconnect.pickle'
    if not os.path.exists(path):
        recalc = True
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    
    if full_connect == False:
        X_embedded = SpectralEmbedding(n_components = k, \
                                       affinity='precomputed_nearest_neighbors',\
                                       n_neighbors = n_neighbors).fit_transform(X)
    else:
        X = np.exp(-X/sigma)
        X_embedded = SpectralEmbedding(n_components = k, \
                                       affinity='precomputed',\
                                       ).fit_transform(X)
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded

def Isomap_embedding(X, k, n_neighbors, loc=None,\
                       used_method = 'IC1CS2SS7',recalc = True):
    path = loc+used_method+'Isomap_k'+str(k)+\
                'neighbor'+str(n_neighbors)+'.pickle'
    if not os.path.exists(path):
        recalc = True
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    
    X_embedded = Isomap(n_components = k, \
                        metric='precomputed',\
                        n_neighbors = n_neighbors).fit_transform(X)
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded

"""
PCA as a benchmark
"""
def PCA_embedding(k, loc=None, recalc = True):
    path = loc + 'PCA_k' + str(k) + '.pickle'
    if os.path.exists(path):
        recalc = False
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    with open(loc + 'OneHot.pickle','rb') as f:
        X_OneHot = pickle.load(f)
    transformer = PCA(n_components=k,
                      random_state=0)
    # Standardizing the features
    X_scaled = StandardScaler().fit_transform(X_OneHot)
    X_embedded = transformer.fit_transform(X_scaled) 
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded

"""
truncated PCA as a benchmark
"""
def tPCA_embedding(k, loc=None, recalc = True):
    path = loc + 'tPCA_k' + str(k) + '.pickle'
    if os.path.exists(path):
        recalc = False
    if recalc == False:
        X_embedded = load_pickle(path)
        return X_embedded
    with open(loc + 'OneHot.pickle','rb') as f:
        X_OneHot = pickle.load(f)
    transformer = TruncatedSVD(n_components=k,
                      random_state=0)
    # Standardizing the features
    X_embedded = transformer.fit_transform(X_OneHot) 
    if loc != None:
        dump_pickle(path, X_embedded)
    return X_embedded