#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 21:31:25 2020
When run in HPC with len(sys.argv) == 3, split all jobs into n sub-jobs, run k th job in this script.
# argv: k (k th job, k = 1,2,...,n), n (total number of job) 
@author: wsd
"""

import os, re, sys
import pickle
import time
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import multiprocessing as mp
# import xgboost as xgb

NUM_CPU = mp.cpu_count()
"""
load non-embedded features
"""
with open('../data/tmp/processed_features/processed_features.pickle','rb') as f:
    (X_BODY, X_CHRON, X_CATE, X_NUM, y) = pickle.load(f)
    
"""
Prepare an empty dataframe to record performance and settings of embeddings and predictions
"""
metric_record = pd.DataFrame(columns = ['Concept_lvl','Set_lvl',
                                        'DR_method','Hyper','Dimension',
                                        'Classifier','Time_used',
                                        'Embedding_only','AUC'])
if not os.path.exists('../results/prediction_metric.csv'):
    metric_record.to_csv('../results/prediction_metric.csv',index=False)
"""
Repeat predicion task for each embedding that is stored in '../data/tmp/embedding'
When run in HPC with len(sys.argv) == 3, split all jobs into n sub-jobs, run k th job in this script.
# argv: k (k th job), n (total number of job) 
"""
all_embedding_files = sorted(os.listdir("../data/tmp/embedding"))
if len(sys.argv) == 3:
    # argv: k (k th job), n (total number of job) 
    k = int(sys.argv[1])
    n = int(sys.argv[2])
    N = len(all_embedding_files)
    idx_from = list(range(0, N, N//n))[k-1]
    if k != n:
        idx_to = list(range(0, N, N//n))[k]
    else:
        idx_to = N
    all_embedding_files = all_embedding_files[idx_from : idx_to]
    
for file in all_embedding_files:
    if not file.endswith(".pickle"):
        continue
    if os.path.exists('../results/predicted_y/'+'1_LR_'+file[:-7]+'.csv'):
        continue
    """
    Hyper parameters of dimension reduction, distance methods are included in 
    file names of embedding.pickle. Extrac them here and then record in the 
    final dataframe of performance.
    """
    if re.search(r'Cosine', file):
        concept_lvl = 'new'
    elif re.search(r'IC1CS2', file):
        concept_lvl = 'WP'
    else:
        concept_lvl = 'n.a.'
    if re.search('SS(\d)', file):
        set_lvl = 'SD' + str( int(re.search('SS(\d)', file).group(1))-4 )
    else:
        set_lvl = 'n.a.'
    if re.search(r'Eigenmap', file):
        DR_method = 'LE'
    elif re.search(r'UMAP', file):
        DR_method = 'UMAP'
    elif re.search(r'OneHot', file):
        DR_method = 'OneHot'
    elif re.search(r'tPCA', file):
        DR_method = 'tPCA'
    else:
        DR_method = 'PCA'
    if re.search('_k\d+(.+)\.pickle', file):
        hyper = re.search('_k\d+(.+)\.pickle', file).group(1)
    else:
        hyper = 'n.a.'
        
    """
    Load embedding 
    """
    print('processing',file[:-7],'...')
    path = os.path.join("../data/tmp/embedding", file)
    with open(path, 'rb') as f:
        X_embedded = pickle.load(f)
    
    """
    Scale embedding before prediction
    """
    if file != "OneHot.pickle":
        X_embedded = StandardScaler().fit_transform(X_embedded)
    dim = X_embedded.shape[1]
    
    """
    Srepare classifiers
    """
    LR = LogisticRegression(penalty='l1', C=0.1,n_jobs=NUM_CPU-2,class_weight='balanced',
                          max_iter = 40000, solver='saga',random_state=0)
    AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
    RF = RandomForestClassifier(max_depth=3,n_jobs=NUM_CPU-2, random_state=0,class_weight='balanced')
    # SupportVC = SVC(kernel='linear',probability=True,class_weight='balanced')
    
    """
    Cross validation for each classifier
    """
    for clf, clf_name in zip([LR,AdaBoost,RF],['LR','AdaBoost','RF']):
        """
        predict with generated embeddings and other features
        """
        X = np.concatenate([X_embedded, X_BODY, X_CHRON, X_CATE, X_NUM],axis=1)
        start = time.time()
        cv = StratifiedKFold(5, shuffle=True, random_state=0) 
        pred = cross_val_predict(clf,X,y,cv=cv, method='predict_proba')
        end = time.time()
        print('prediction time used:',(end-start)/3600,'H')
        # 0 indicate NOT embedding only
        df0 = pd.DataFrame({'pred_prob':pred[:,1],'true_y':y})
        df0.to_csv('../results/predicted_y/'+'0_'+clf_name+'_'+file[:-7]+'.csv',index=False)
        auc = roc_auc_score(y,pred[:,1])
        tmp_df = pd.DataFrame({'Concept_lvl':concept_lvl,'Set_lvl':set_lvl,
                            'DR_method':DR_method,'Hyper':hyper,'Dimension':dim,
                            'Classifier':clf_name,'Time_used': end-start,
                            'Embedding_only':0,'AUC':auc},index=[0])
        tmp_df.to_csv('../results/prediction_metric.csv',index=False,mode='a', header=False)
        metric_record = metric_record.append(tmp_df, ignore_index=True)
        # metric_record.loc[-1] = [file[:-7],0,auc]
        # metric_record.reset_index(drop=True,inplace=True)
        """
        predict with only embeddings
        """
        cv = StratifiedKFold(5, shuffle=True, random_state=0) 
        pred = cross_val_predict(clf, X_embedded,y,cv=cv, method='predict_proba')
        # 1 indicate embedding only
        df1 = pd.DataFrame({'pred_prob':pred[:,1],'true_y':y})
        df1.to_csv('../results/predicted_y/'+'1_'+clf_name+'_'+file[:-7]+'.csv',index=False)
        auc = roc_auc_score(y,pred[:,1])
        tmp_df = pd.DataFrame({'Concept_lvl':concept_lvl,'Set_lvl':set_lvl,
                            'DR_method':DR_method,'Hyper':hyper,'Dimension':dim,
                            'Classifier':clf_name,'Time_used': end-start,
                            'Embedding_only':1,'AUC':auc},index=[0])
        tmp_df.to_csv('../results/prediction_metric.csv',index=False,mode='a', header=False)
        metric_record = metric_record.append(tmp_df, ignore_index=True)
        # metric_record.loc[-1] = [file[:-7],1,auc.mean()]
        # metric_record.reset_index(drop=True,inplace=True)
        if 'k' in locals():
            metric_record.to_csv('../results/prediction_metric_part'+str(k)+'.csv',index=False)
        else:
            metric_record.to_csv('../results/prediction_metric.csv',index=False)
    
