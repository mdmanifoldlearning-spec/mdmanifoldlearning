# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:00:41 2021

@author: Shaodong
"""

import os, re, sys, argparse
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
from eval_tools import cv_evaluation, nested_cv_eval
from data_processing_tools import prepare_nrd_data, prepare_mimic_data, get_target_label
from utils import load_pickle, dump_pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_name", type=str, choices=['nrd', 'mimic'],
                    help="data name", required=True)
parser.add_argument("-s", "--selected_disease", type=str, 
                    help="selected disease in mimic dataset", required=True)
parser.add_argument("-t", "--task_name", type=str, 
                    choices=['readmission', 'mortality'],
                    help="task name, readmission or mortality", required=True)
parser.add_argument("-e", "--embedding_only", type=str, 
                    choices=['True', 'False'],
                    help="if use embedding only for prediction", required=True)
args = parser.parse_args()

data_name = args.data_name
selected_disease = args.selected_disease
task_name = args.task_name
embedding_only = args.embedding_only == 'True'


if data_name == 'mimic':
    data, icd_cols, demo_cols = prepare_mimic_data(selected_disease)
    csv_file_name = '/MD_manifold_cv.csv' if embedding_only else '/MD_manifold_morefeature_cv_test.csv'
    embedding_dir = '../data/tmp/'+data_name+'_'+selected_disease+'/'+task_name+'/embedding/'
    result_path = '../results/'+data_name+'_'+selected_disease+'/'+task_name+csv_file_name
    X_demo = data[demo_cols]
else:
    data, icd_cols, _, demo_cols = prepare_nrd_data()
    csv_file_name = '/MD_manifold_cv.csv' if embedding_only else '/MD_manifold_morefeature_cv.csv'
    embedding_dir = '../data/tmp/nrd/'+task_name+'/embedding/'
    result_path = '../results/nrd/'+task_name+csv_file_name
    X_demo = data[demo_cols]
    
embedding_path_list = os.listdir(embedding_dir)
embedding_path_list = [p for p in embedding_path_list if '.pickle' in p]

y_mortality, y_readmission = get_target_label(data)
X_demo = X_demo.fillna(X_demo.mean(numeric_only=True))
X_demo = X_demo[y_mortality == 0].values if task_name == 'readmission' else X_demo.values
y = y_mortality if task_name == 'mortality' else y_readmission

for embedding_path in embedding_path_list:
    print('processing', embedding_path)
    if os.path.exists(result_path):
        result_df = pd.read_csv(result_path)
    else:
        result_df = pd.DataFrame()
    embedding= load_pickle(embedding_dir+embedding_path)
    X = embedding if embedding_only else np.concatenate((embedding, X_demo), axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if np.isnan(embedding).any():
        print('nan in', embedding_path)
        continue
    cv_evaluation(X=X, y=y, task_name=task_name, 
                  data_name=data_name, model_name=embedding_path[:-7], 
                  res_save_path = result_path)




