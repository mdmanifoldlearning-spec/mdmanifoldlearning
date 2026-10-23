# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:18:54 2021

@author: Shaodong
"""
from traditional_baseline_models import (
    replace_code,
    dic_icd9_ccs,
    dic_icd9_cui,
    dic_icd9_snomed,
    risk_index,
    cci_codes, cci_weights,
    eci_codes, eci_weights,
    rsi_codes, rsi_weights, rsi_b
)
from utils import save_to_df

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

import numpy as np
import pandas as pd
import time, os
import itertools


def cv_evaluation(X, y, task_name, data_name, model_name, res_save_path, seed=0):
    def check_if_evaluated(params):
        if not os.path.exists(res_save_path):
            return False
        result_df = pd.read_csv(res_save_path)
        cols_to_check = ['model_name', 'task_name', 'dim']
        vals_to_check = (model_name, task_name, dim)
        if 'params' in result_df.columns:
            cols_to_check = cols_to_check + ['params']
            vals_to_check = vals_to_check + (str(params), )
        if seed != 0:
            cols_to_check = cols_to_check + ['seed']
            vals_to_check = vals_to_check + (seed, )
            
        has_evaluated = (result_df[cols_to_check] == vals_to_check).all(1).any()

        # if 'params' in result_df.columns:
        #     has_evaluated = (result_df[['model_name', 'task_name', 'dim', 'params']] == (model_name, task_name, dim, str(params))).all(1).any()
        # else:
        #     has_evaluated = (result_df[['model_name', 'task_name', 'dim']] == (model_name, task_name, dim)).all(1).any()
        return has_evaluated

    dim = X.shape[1]
    # LR = LogisticRegression(penalty='l1', C=1,class_weight='balanced',
    #                         max_iter = 40000, solver='saga',random_state=0)
    # AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
    # RF = RandomForestClassifier(max_depth=3, random_state=0,class_weight='balanced')
    # xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
    # NN = MLPClassifier(solver='adam', alpha=1e-5, random_state=0)
    # XGB_params = {'max_depth':[3, 6, 9], 'colsample_bytree': [0.5, 0.8], 'alpha': [0.1, 0.5]}
    XGB_params = {'max_depth':[9], 'colsample_bytree': [0.8], 'alpha': [0.5]}
    LR_params = {'C': [1, 0.1]}
    NN_params = {'hidden_layer_sizes': [(256, 128)]} #, (256, 64), (256, 128, 64)
    all_params = {'XGBoost': XGB_params, 'LR': LR_params, 'NN': NN_params}
    # for clf_name in ['XGBoost', 'LR', 'NN']:
    for clf_name in ['XGBoost']:
    # for clf, clf_name in zip([LR, AdaBoost, RF], ['LR','AdaBoost', 'RF']):
        for params in get_args_combinations(all_params[clf_name]):
            if check_if_evaluated(params) == True:
                continue
            if clf_name == 'XGBoost':
                clf = xgb.XGBClassifier(**params, objective="binary:logistic", random_state=0)
            elif clf_name == 'LR':
                clf = LogisticRegression(**params, penalty='l1', class_weight='balanced',
                    max_iter = 40000, solver='saga',random_state=0)
            elif clf_name == 'NN':
                clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=2000, random_state=0)

            cv = StratifiedKFold(5, shuffle=True, random_state=seed) 
            pred = cross_val_predict(clf, X, y, cv=cv, method='predict_proba', n_jobs = -1)
            auc = roc_auc_score(y, pred[:,1])
            print(seed, data_name, model_name, task_name, dim, clf_name, auc, sep=',')
            perf_to_save = {'seed':seed,'data_name':data_name, 'auc':auc, 
                            'model_name':model_name, 'task_name':task_name, 
                            'dim':dim, 'clf_name':clf_name, 'params': str(params)}
            save_to_df(perf_to_save, res_save_path)

def get_args_combinations(args_to_search):
    '''
    Params:
        args_to_search: dict, args name (str) -> list of args
    Return:
        args_list: list of dict, which maps args name (str) to args value (scalar)
    '''
    param_list = list(args_to_search.keys())
    args_to_search_list = [args_to_search[param] for param in param_list]
    args_combination_list = list(itertools.product(*args_to_search_list))
    args_list = [dict(zip(param_list, args)) for args in args_combination_list]
    return args_list

def cv_evaluation_time(X, y, task_name, data_name, model_name, res_save_path):
    dim = X.shape[1]
    LR = LogisticRegression(penalty='l1', C=1,class_weight='balanced',
                            max_iter = 40000, solver='saga',random_state=0)
    AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=0)
    RF = RandomForestClassifier(max_depth=3, random_state=0,class_weight='balanced')
    for clf, clf_name in zip([LR], ['LR']):
    # for clf, clf_name in zip([AdaBoost, RF], ['AdaBoost', 'RF']):
        cv = StratifiedKFold(5, shuffle=True, random_state=0) 
        # scores = cross_validate(clf, X, y, scoring = 'roc_auc')
        # fit_time = np.sum(scores['fit_time'])
        # # pred = cross_val_predict(clf, X, y, cv=cv, method='predict_proba', n_jobs = -1)
        # # auc = roc_auc_score(y, pred[:,1])
        # print('fit_time:', data_name, model_name, task_name, dim, clf_name, fit_time, sep=',')
        # perf_to_save = {'data_name':data_name, 'fit_time':fit_time, 
        #                 'model_name':model_name, 'task_name':task_name, 
        #                 'dim':dim, 'clf_name':clf_name}
        # if res_save_path is not None:
        #     save_to_df(perf_to_save, res_save_path)
        start = time.time()
        pred = cross_val_predict(clf, X, y, cv=cv, method='predict_proba', n_jobs = -1)
        cv_time = time.time() - start
        auc = roc_auc_score(y, pred[:,1])
        print('cv_time:', data_name, model_name, task_name, dim, clf_name, auc, cv_time, sep=',')
        
            
def index_evaluation(
        icd_df, y_mortality, y_readmission, data_name, res_save_path, 
        embedding_only, X_demo_readmission = None, X_demo_mortality=None):
    for X_demo, y, task_name in zip(
            [X_demo_mortality, X_demo_readmission], 
            [y_mortality, y_readmission], 
            ['mortality', 'readmission']):
        if task_name == 'readmission':
            icd_df_eval = icd_df.iloc[y_mortality.to_numpy()==0,:]
        else:
            icd_df_eval = icd_df
            
        cci = risk_index(icd_df_eval, cci_codes, cci_weights)
        eci = risk_index(icd_df_eval, eci_codes, eci_weights)
        rsi = risk_index(icd_df_eval, rsi_codes, rsi_weights, b=rsi_b)
        if embedding_only:
            auc_cci = roc_auc_score(y, cci)
            auc_eci = roc_auc_score(y, eci)
            auc_rsi = roc_auc_score(y, rsi)
            print(data_name, task_name, 'cci', 'auc', auc_cci, sep=',')
            print(data_name, task_name, 'eci', 'auc', auc_eci, sep=',')
            print(data_name, task_name, 'rsi', 'auc', auc_rsi, sep=',')
            save_to_df({'data_name':data_name, 'task_name':task_name, 'model_name':'cci', 'auc': auc_cci}, res_save_path)
            save_to_df({'data_name':data_name, 'task_name':task_name, 'model_name':'eci', 'auc': auc_eci}, res_save_path)
            save_to_df({'data_name':data_name, 'task_name':task_name, 'model_name':'rsi', 'auc': auc_rsi}, res_save_path)
        else:
            assert X_demo is not None
            X_cci = cci.values.reshape((len(cci),1))
            X_eci = eci.values.reshape((len(eci),1))
            X_rsi = rsi.values.reshape((len(rsi),1))
            X_cci_demo = np.concatenate((X_cci, X_demo), axis=1)
            X_eci_demo = np.concatenate((X_eci, X_demo), axis=1)
            X_rsi_demo = np.concatenate((X_rsi, X_demo), axis=1)
            cv_evaluation(X_cci_demo, y, task_name, data_name, 'cci', res_save_path)
            cv_evaluation(X_eci_demo, y, task_name, data_name, 'eci', res_save_path)
            cv_evaluation(X_rsi_demo, y, task_name, data_name, 'rsi', res_save_path)
            
def nested_cv_eval(X, y, info_to_save_dict):
    """
    Nested cross validation. Search optimal hyper-parameter in the inner CV.
    classifiers: list 
    param_grids: list of dict of parameters. Note this list must be consistent with "classifiers".
    X: numpy array
    y: numpy array
    inner_cv: cross-validation module in sklearn.model_selection
    outer_cv: cross-validation module in sklearn.model_selection
    """
    outer_cv = StratifiedKFold(5, shuffle=True, random_state=0) 
    inner_cv = StratifiedKFold(3, shuffle=True, random_state=0) 
    classifiers, param_grids = get_classifiers_and_params()
    sample_size = get_sample_size(X)
    feature_size = get_feature_size(X)
    output = pd.DataFrame()
    for clf, p_grid in zip(classifiers, param_grids):
        start_time = time.time()
        # Grid Search in the inner CV
        search = GridSearchCV(estimator=clf, param_grid=p_grid, cv=inner_cv, scoring = 'roc_auc', refit = True) 
        # prediction in the outter CV. Note "search" will fit with the training data of outer_cv.
        pred = cross_val_predict(search, X=X, y=y, cv=outer_cv, method='predict_proba', n_jobs=-1) 

        perf = performance(y, pred[:,1])
        perf_str = str(perf)
        clf_name = classifier_name(clf)
        p_grid_str = str(p_grid)
        end_time = time.time()
        time_used = (end_time - start_time) / 60.0
        
        # print('Classifier:', clf_name, '\nparameters searched', p_grid_str, 
        #       '\nperformance', perf_str, '\nsample size:',sample_size, 
        #       'TimeUsed(m)', time_used, '\n------------')
        perf.update({'clf_name': clf_name, 'parameters_searched': p_grid_str, 
                     'sample_size': sample_size, 'dim': feature_size, 'time_used(m)': time_used})
        perf.update(info_to_save_dict)

        output = output.append(perf, ignore_index=True)
    return output

def get_classifiers_and_params():
    RF = RandomForestClassifier(random_state=0, n_estimators = 800, class_weight='balanced')
    LR = LogisticRegression(penalty='l1', class_weight='balanced',
                max_iter = 40000, solver='saga',random_state=0)
    LinearSVC = SVC(kernel='linear', class_weight='balanced', max_iter = 80000, probability = True, random_state=0)
    AdaBoost = AdaBoostClassifier(random_state=0)
    MLP = MLPClassifier(random_state=0, max_iter=3000)
    classifiers = [RF, LR, LinearSVC, AdaBoost, MLP]

    param_RF = {'max_depth':[2, 3, 4, 8]}
    param_LR = {'C':[0.5, 1, 2]}
    param_SVC = {'C': [0.05, 0.1, 0.5, 1]}
    param_AdaBoost = {'n_estimators': [100, 200, 400, 800]}
    param_MLP = {'hidden_layer_sizes': [(100, 50, 30),(30,20,10),(100,30),(50, 30), (30,10), (50,), (30,)]}
    param_grids = [param_RF, param_LR, param_SVC, param_AdaBoost, param_MLP]

    assert len(classifiers) == len(param_grids)
    return classifiers, param_grids

def get_sample_size(data_set):
    """
    sample_size 
    return: number of rows 
    """
    return data_set.shape[0]

def get_feature_size(data_set):
    return data_set.shape[1]

def performance(y_true, y_pred):
    """
    return: classification performance: dic 
    """
    perf = dict()
    metrics = [roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support]
    for m in metrics:
        if m.__name__ == 'precision_recall_fscore_support':
            try:
                scores = m(y_true, y_pred)
            except:
                scores = m(y_true, y_pred>=0.5)
            scores_str = [' & '.join(s.round(3).astype(str)) for s in scores]
            perf.update(dict(zip(['precision', 'recall', 'fbeta_score', 'support'], scores_str)))
        else:
            try:
                perf[m.__name__] = m(y_true, y_pred)
            except:
                perf[m.__name__] = m(y_true, y_pred>=0.5)
    return perf

def classifier_name(classifier):
    """
    return: classifier's name: str
    -------
    todo: padding classifier's major parameters
    """
    return classifier.__class__.__name__

def check_evaluated(perf_df, cols, col_vals):
    if len(perf_df.index) == 0:
        return False
    check_each_row = (perf_df[cols] == col_vals).all(axis=1)
    return check_each_row.any()    