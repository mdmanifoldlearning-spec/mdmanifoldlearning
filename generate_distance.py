#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from icd9cms.icd9 import search
from distance import ConceptLevelDistance, get_all_visit_dist
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import argparse
from traditional_baseline_models import dic_icd9_ccs, dic_icd9_cui, dic_icd9_snomed
from data_processing_tools import prepare_nrd_data, prepare_mimic_data, get_data_for_task
from utils import dump_pickle, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_name", type=str, choices=['nrd', 'mimic'],
                    help="data name", required=True, default = 'mimic')
parser.add_argument("-s", "--selected_disease", type=str, default = '0389',
                    help="selected disease in mimic dataset", required=True)
parser.add_argument("-i", "--icd_method", type=str, default = 'eHDN',
                    choices=['IC1CS2',
                             'Cosine','Group_plus_Cosine',
                             "Euclidean", "Group_plus_Euclidean",
                             "Manhattan", "Group_plus_Manhattan",
                             "eHDN", "Group_plus_eHDN"],
                    help="icd level distance method", required=True)
parser.add_argument("-g", "--group_name", type=str, default = 'None',
                    choices=['ccs', 'cui', 'snomed', 'None'],
                    help="group name, e.g. ccs", required=True)
parser.add_argument("-v", "--visit_method", type=str, default = 'SS7',
                    choices=['SS5','SS6','SS7','SS8'],
                    help="visit level distance method", required=True)
parser.add_argument("-t", "--task_name", type=str, default = 'readmission',
                    choices=['readmission', 'mortality'],
                    help="task name, readmission or mortality", required=True)
parser.add_argument("-n", "--n_jobs", type=int, default = 3,
                    help="number of jobs to submit in multiprocess", required=True)
args = parser.parse_args()
assert args.icd_method != 'Group_plus_Cosine' or args.group_name in ['ccs', 'cui', 'snomed']

data_name = args.data_name
icd_method = args.icd_method
group_name = args.group_name
visit_method = args.visit_method
n_jobs = args.n_jobs
selected_disease = args.selected_disease
task_name = args.task_name

if data_name == 'nrd':
    icd_loc = '../data/tmp/'+data_name+'/'+task_name+'/icd_distance/'
    visit_loc='../data/tmp/'+data_name+'/'+task_name+'/visit_distance/'
else:
    folder_name = '../data/tmp/'+data_name + '_' + selected_disease+'/'+task_name
    icd_loc = folder_name+'/icd_distance/'
    visit_loc = folder_name+'/visit_distance/'

"""
load tree information
"""
all_ICD = search('n/a').descendants() + ['root'] 
n_icds = len(all_ICD)

if group_name == 'ccs':
    ances_dic = dic_icd9_ccs
elif group_name == 'cui':
    ances_dic = dic_icd9_cui
elif group_name == 'snomed':
    ances_dic = dic_icd9_snomed
else:
    ances_dic = {
        e:search(e).ancestors()+['root'] if e != 'root' 
        else [] for e in all_ICD
    }

if group_name in ['ccs', 'cui', 'snomed']:
    all_groups = list({v for _, v in ances_dic.items()})
    all_codes = all_ICD + all_groups
    code2idx = dict(zip(all_codes, range(len(all_codes))))
    codeidx2ancesidx = {code2idx[code]:[code2idx[ances]] for code, ances in ances_dic.items() if code in all_codes}
else:
    all_codes = all_ICD
    code2idx = dict(zip(all_codes, range(len(all_codes))))
    codeidx2ancesidx = {code2idx[code]:[code2idx[c] for c in ances_list] for code, ances_list in ances_dic.items() if code in all_codes}

if __name__ == '__main__':
    print(args)
    """
    load data
    """
    if data_name == 'nrd':
        data, icd_cols, _, demo_cols = prepare_nrd_data()
    elif data_name == 'mimic':
        data, icd_cols, demo_cols = prepare_mimic_data(selected_disease)
    
    data = get_data_for_task(data, task_name)
    # data, icd_cols = data[0:1000], icd_cols[0:1000]
    
    ICD_df = data[icd_cols]

    code_idx_sets = ICD_df.apply(
        lambda each_visit: {code2idx[code] for code in each_visit.dropna()}, axis = 1)

    """
    calculate distance with module distance
    """
    code_distance = ConceptLevelDistance(
        all_codes = all_codes,
        codeidx2ancesidx = codeidx2ancesidx,
        hierarchy_name = group_name
    )
    
    code_dist,_,_ = code_distance.get_all_codes_dist(
        code_idx_sets= code_idx_sets, 
        code_method=icd_method,
        code_loc=icd_loc, 
        code_recalc=True)
    
    visit_dist = get_all_visit_dist(
        code_idx_sets = code_idx_sets, 
        code_dist = code_dist[0:n_icds, 0:n_icds], 
        concept_level_method = icd_method,
        hierarchy_name = group_name,
        set_level_method = visit_method, 
        n_jobs = n_jobs, 
        visit_loc = visit_loc)
    
    







