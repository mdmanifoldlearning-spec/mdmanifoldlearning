# -*- coding: utf-8 -*-

import os, re, sys, argparse, string, nltk
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
from eval_tools import cv_evaluation, nested_cv_eval, get_args_combinations
from data_processing_tools import prepare_nrd_data, prepare_mimic_data, get_target_label, get_mimic_notes
from utils import load_pickle, dump_pickle
from EHR_classification_model import EHRClassifier, Run
import warnings
warnings.filterwarnings('ignore')
import torch
torch.manual_seed(0)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--selected_disease", type=str,
                    help="selected disease in mimic dataset", required=True)
parser.add_argument("-t", "--task_name", type=str,
                    choices=['mortality', 'readmission'],
                    help="if use embedding for prediction", required=True)
parser.add_argument("-u", "--use_embedding", type=str,
                    choices=['True', 'False'],
                    help="if use embedding for prediction", required=True)
parser.add_argument("-b", "--is_baseline", type=str,
                    choices=['True', 'False'],
                    help="if is_baseline ", required=True)
args = parser.parse_args()

selected_disease = args.selected_disease
use_embedding = args.use_embedding == 'True'
is_baseline = args.is_baseline == 'True'

data_name = 'mimic'
task_name = args.task_name
fixed_seed = True
if is_baseline == True:
    assert use_embedding == True

data, icd_cols, tabular_feature_cols = prepare_mimic_data(selected_disease, no_readmit=False, 
    ts_feature=True, text_feature=True)

if is_baseline == False:
    csv_file_name = '/EHR_classification.csv'
    embedding_dir = '../data/tmp/'+data_name+'_'+selected_disease+'/'+task_name+'/embedding/'
else:
    csv_file_name = '/EHR_classification_baseline.csv'
    embedding_dir = '../data/tmp/'+data_name+'_'+selected_disease+'/'+task_name+'/baseline_visit_embedding/'
result_path = '../results/'+data_name+'_'+selected_disease+'/'+task_name+csv_file_name
X_tabular = data[tabular_feature_cols]
X_ts = data['TS']
X_text = data['TEXT']
    
embedding_path_list = os.listdir(embedding_dir)
embedding_path_list = [p for p in embedding_path_list if '.pickle' in p]
if is_baseline:
    embedding_path_list = [p for p in embedding_path_list if 'GloVe' in p or '3digit' in p]
else:
    # embedding_path_list = [p for p in embedding_path_list if 'neighbor16' in p or 'neighbor32' in p]
    embedding_path_list = [p for p in embedding_path_list if 'Cosine' in p and 'SS6' in p]

y_mortality, y_readmission = get_target_label(data)
X_tabular = X_tabular.fillna(X_tabular.mean(numeric_only=True))
X_tabular = X_tabular[y_mortality == 0] if task_name == 'readmission' else X_tabular
X_text = X_text[y_mortality == 0] if task_name == 'readmission' else X_text
X_ts = X_ts[y_mortality == 0] if task_name == 'readmission' else X_ts
y = y_mortality if task_name == 'mortality' else y_readmission

# %% preprocess text data
import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format(
     '../data/mimic-iii/BioWordVec_PubMed_MIMICIII_d200.vec.bin',
      binary=True,
      # limit=None, # this fuckin thing has 4 billion tokens (4E9)
      limit=int(1E4) # faster load if you limit to most frequent terms
)

stop_words = nltk.corpus.stopwords.words('english')
def preprocess_notes(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in nltk.word_tokenize(text) \
        if token not in string.punctuation and token not in stop_words]

    return tokens
X_text = X_text.apply(preprocess_notes)

def token_to_index(tokens_list, word2vec):
    index_list = [word2vec.key_to_index[words] for words in tokens_list if words in word2vec.key_to_index]
    return index_list
X_text = X_text.apply(lambda notes: token_to_index(notes, word2vec))

# padding 0
max_len = X_text.apply(len).max()
print('max seq len:', max_len)
def padding_to_equal_len(tokens_list, max_len):
    pad_idx = 0
    len_to_pad = max_len - len(tokens_list)
    list_to_pad = [pad_idx] * len_to_pad
    return tokens_list + list_to_pad
X_text = X_text.apply(lambda tokens_list: padding_to_equal_len(tokens_list, max_len))
# %%
def cv_split(X_text, X_ts, X_tabular, y, md_manifold_y_pred=None, seed=0):
    stratify_cv = StratifiedKFold(
        n_splits=5,  shuffle=True, random_state=seed)
    split_generator = stratify_cv.split(X_text, y)
    for train_idx, test_idx in split_generator:
        # split train / test
        X_text_train, X_text_test = X_text.iloc[train_idx], X_text.iloc[test_idx]
        X_ts_train, X_ts_test = X_ts.iloc[train_idx], X_ts.iloc[test_idx]
        X_tabular_train, X_tabular_test = X_tabular.iloc[train_idx], X_tabular.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # pd to np.array
        X_text_train, X_text_test = np.array(X_text_train.tolist()), np.array(X_text_test.tolist())
        X_ts_train, X_ts_test = np.stack(X_ts_train.tolist()).astype(float), np.stack(X_ts_test.tolist()).astype(float)
        X_tabular_train, X_tabular_test = X_tabular_train.values.astype(float), X_tabular_test.values.astype(float)
        y_train, y_test = y_train.values, y_test.values

        if md_manifold_y_pred is not None:
            md_manifold_y_pred_test = md_manifold_y_pred[test_idx]
        else:
            md_manifold_y_pred_test = None

        data = {'x_train': tuple(zip(X_text_train, X_ts_train, X_tabular_train)), 'y_train': y_train,\
            'x_test': tuple(zip(X_text_test, X_ts_test, X_tabular_test)), 'y_test': y_test}

        yield data, md_manifold_y_pred_test

def get_prob_in_cv(tabular_feature, y, seed):
    clf = LogisticRegression(penalty='l1', C=1, class_weight='balanced',
        max_iter = 40000, solver='saga',random_state=0)
    stratify_cv = StratifiedKFold(
        n_splits=5,  shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, tabular_feature, y, cv=stratify_cv, method='predict_proba', n_jobs = -1)
    return pred[:,1]

def cv_evaluation_NN(X_text, X_ts, X_tabular, y, params, md_manifold_y_pred, seed):
    perf_list = []
    for cv_i, (data, md_manifold_y_pred_test) in enumerate(cv_split(X_text, X_ts, X_tabular, y, md_manifold_y_pred, seed)):
        nn_model = EHRClassifier(params=params, word2vec=word2vec)
        perf = Run.train(nn_model, data, params, md_manifold_y_pred_test)
        perf['cv'] = cv_i
        perf_list.append(perf)
    perf_df = pd.concat(perf_list, ignore_index=True)
    return perf_df
    # if 'ensemble_a' in perf_df.columns:
    #     cv_perf_avg = perf_df.groupby('ensemble_a').mean().reset_index()
    # else:
    #     cv_perf_avg = perf_df.mean().to_frame().transpose()
    # return cv_perf_avg

def check_evaluated(perf_df, cols, col_vals):
    if len(perf_df.index) == 0:
        return False
    check_each_row = (perf_df[cols] == col_vals).all(axis=1)
    return check_each_row.any()    

#%%
params_to_search = {
    'device': ['cuda'],
    'word_embedding_size': [200],
    'tabular_feature_size': [X_tabular.shape[1]],
    'rnn_input_dim': [X_ts.iloc[0].shape[1]],
    'rnn_hidden_dim': [256], 
    'rnn_num_layers': [4],
    'cnn_out_size': [256],
    'fcn_hidden_dim': [128, 256],
    'stride': [1],
    'lr': [1e-5],
    'batch_size': [32],
    'epochs': [16]
}


if use_embedding == False:
    if os.path.exists(result_path):
        perf_df = pd.read_csv(result_path)
    else:
        perf_df = pd.DataFrame()
    for params in get_args_combinations(params_to_search):
        has_evaluated = check_evaluated(perf_df, ['params_str'], \
            (str(params)))
        if has_evaluated:
            continue
        cv_perf_avg = cv_evaluation_NN(X_text, X_ts, X_tabular, y, params, None, 0)
        cv_perf_avg['manifold_embedding'] = 'None'
        cv_perf_avg['params_str'] = str(params)
        for k, v in params.items():
            cv_perf_avg[k] = v
        perf_df = pd.concat([perf_df, cv_perf_avg])
        perf_df.to_csv(result_path, index=False)

if use_embedding == True:
    if os.path.exists(result_path):
        perf_df = pd.read_csv(result_path)
    else:
        perf_df = pd.DataFrame()
    for embedding_path in embedding_path_list:
        embedding= load_pickle(embedding_dir+embedding_path)
        X_tabular_embedding = pd.concat([X_tabular, pd.DataFrame(embedding)], axis=1)
        for params in get_args_combinations(params_to_search):
            seed_options = [0] if fixed_seed == True else range(1, 10)
            for seed in seed_options:
                # try:
                if fixed_seed == False:
                    has_evaluated = check_evaluated(perf_df, \
                        ['params_str', 'manifold_embedding', 'seed'], \
                            (str(params), embedding_path[:-7], seed))
                else:
                    has_evaluated = check_evaluated(perf_df, \
                        ['params_str', 'manifold_embedding'], \
                            (str(params), embedding_path[:-7]))
                if has_evaluated:
                    continue

                # md_manifold_y_pred = get_prob_in_cv(embedding, y, seed=seed)
                cv_perf_avg = cv_evaluation_NN(X_text, X_ts, X_tabular_embedding, y=y, 
                    params=params, md_manifold_y_pred=None, seed=seed)
                cv_perf_avg['manifold_embedding'] = embedding_path[:-7]
                cv_perf_avg['params_str'] = str(params)
                if fixed_seed == False:
                    cv_perf_avg['seed'] = seed
                for k, v in params.items():
                    cv_perf_avg[k] = v
                perf_df = pd.concat([perf_df, cv_perf_avg])
                perf_df.to_csv(result_path, index=False)





# %%
