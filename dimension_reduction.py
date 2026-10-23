#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:56:10 2020
notes: we can calculate all distances for the whole dataset, then use 
metric =  function to take out the distance. In this way, embedding of new data
can be easy, when we use supervised version of embedding.
@author: wsd
"""
# import umap
import numpy as np
import pickle
import sys, os
import time
import argparse
from utils import load_pickle, dump_pickle
from manifold_models import (
    Eigenmap_embedding,
    # UMAP_embedding,
    TSNE_embedding,
    PCA_embedding,
    tPCA_embedding,
    Isomap_embedding
)



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_name", type=str, choices=['nrd', 'mimic'],
                    help="data name", required=True)
parser.add_argument("-s", "--selected_disease", type=str, 
                    help="selected disease in mimic dataset", required=True)
parser.add_argument("-t", "--task_name", type=str, 
                    choices=['readmission', 'mortality'],
                    help="task name, readmission or mortality", required=True)
parser.add_argument("-i", "--icd_method", type=str, default = 'eHDN',
                    choices=['IC1CS2',
                             'Cosine','Group_plus_Cosine',
                             "Euclidean", "Group_plus_Euclidean",
                             "Manhattan", "Group_plus_Manhattan",
                             "eHDN", "Group_plus_eHDN"],
                    help="icd level distance method", required=True)
parser.add_argument("-v", "--visit_method", type=str, default = 'SS7',
                    choices=['SS5','SS6','SS7','SS8'],
                    help="visit level distance method", required=True)
args = parser.parse_args()

data_name = args.data_name
selected_disease = args.selected_disease
task_name = args.task_name
visit_method = args.visit_method
icd_method = args.icd_method

if data_name == 'nrd':
    embedding_dir = '../data/tmp/'+data_name+'/'+task_name+'/embedding/'
    visit_dist_dir = '../data/tmp/'+data_name+'/'+task_name+'/visit_distance/'
else:
    embedding_dir = '../data/tmp/'+data_name+'_'+selected_disease+'/'+task_name+'/embedding/'
    visit_dist_dir = '../data/tmp/'+data_name+'_'+selected_disease+'/'+task_name+'/visit_distance/'
visit_dist_path_list = os.listdir(visit_dist_dir)
visit_dist_path_list = [visit_dist_dir + p for p in visit_dist_path_list]
visit_dist_path_list = [p for p in visit_dist_path_list if 'pickle' in p]
visit_dist_path_list = [p for p in visit_dist_path_list if icd_method in p]
visit_dist_path_list = [p for p in visit_dist_path_list if visit_method in p]

if __name__ == '__main__':

    for visit_dist_path in visit_dist_path_list:
        visit_dist = load_pickle(visit_dist_path)
        dist_method = visit_dist_path.split('/')[-1][:-7]
        # Laplacian Eigen Map
        if False:
            # k is the size of the embeddings
            for k in [16,32,64,128,256,512]:
                for n_neighbors in [4, 8, 16, 32, 64, 128, 256, 512]:
                    try:
                        start = time.time()
                        X_embedded = Eigenmap_embedding(visit_dist.astype(float),\
                                            k=k,\
                                            n_neighbors = n_neighbors,\
                                            loc = embedding_dir,\
                                            used_method=dist_method,\
                                            recalc = False,\
                                            full_connect=False)
                        end = time.time()
                        print('Eigen Map embedding, k:', k,', n_neighbors:',n_neighbors,', time used(s):',end-start)
                    except:
                        pass
            # if True:
            #     for sigma in [0.1]:
            #         start = time.time()
            #         X_embedded = Eigenmap_embedding(visit_dist.astype(float),\
            #                             k=k,sigma=sigma,\
            #                             n_neighbors = None,\
            #                             loc = embedding_dir,\
            #                             used_method=dist_method,\
            #                             recalc = True,\
            #                             full_connect=True)
            #         end = time.time()
            #         print('Eigen Map embedding full connect, time used(s):',end-start)
        #Isomap
        if True:
            for k in [3]:
                for n_neighbors in [4, 8, 16, 32, 64, 128, 256, 512]:
                    try:
                        start = time.time()
                        X_embedded = Isomap_embedding(visit_dist.astype(float),\
                                            k=k,\
                                            n_neighbors = n_neighbors,\
                                            loc = embedding_dir,\
                                            used_method=dist_method,\
                                            recalc = False)
                        end = time.time()
                        print('Iso Map embedding, k:', k,', n_neighbors:',n_neighbors,', time used(s):',end-start)
                    except:
                        pass
        # tSNE
        if False:
            for k in [2]:
                for perplexity in [10,20,30,40]:
                    start = time.time()
                    X_embedded = TSNE_embedding(visit_dist.astype(float), \
                                            k=k,perplexity = perplexity,\
                                            loc = embedding_dir,\
                                            used_method=dist_method,\
                                            recalc = True)
                    end = time.time()
                    print('tSNE embedding time used(s):',end-start)
        # UMAP
        if False:
            for k in [2, 3]:
                # according to UMAP API doc, neighbors should be 2-100.
                for n_neighbors in [32,128,512,2048,8192,visit_dist.shape[0]-1]:
                    for min_dist in [0.1,0.25,0.5]:
                        start = time.time()
                        X_embedded = UMAP_embedding(visit_dist.astype(float), \
                                        k=k, n_neighbors=n_neighbors,\
                                        min_dist = min_dist,\
                                        loc = embedding_dir,\
                                        used_method=dist_method,\
                                        recalc = True)
                        end = time.time()
                        print('UMap embedding n_neighbors:', n_neighbors, 'time used(s):',end-start)








