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
import os, re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

def get_target_label(df):
    y_mortality = df['DIED']
    y_readmission = df[y_mortality==0]['READMISSION']
    return y_mortality, y_readmission

def prepare_nrd_data():
    """
    load nrd 
    """
    nrd_icd_cols = ['DX'+str(i) for i in range(1, 31)] + ['ECODE'+str(i) for i in range(1,5)]
    nrd_ccs_cols = ['DXCCS'+str(i) for i in range(1, 31)] + ['E_CCS'+str(i) for i in range(1,5)]
    target_cols = ['READMISSION', 'DIED']
    nrd2014 = pd.read_csv('../data/NRD_2014/NRD_2014_S109_HF.csv', 
                          dtype = "string")
    nrd2014['DIED'] = nrd2014['DIED'].astype(int)
    nrd2014['READMISSION'] = nrd2014['READMISSION'].astype(int)
    
    num_feature_cols = ['AGE', 'NCHRONIC','NDX',
                        'NECODE','NPR', 'FEMALE', 'HCUP_ED']
    onehot_data_df, cate_feature_cols = one_hot_with_column_name(
        nrd2014, 
        ['AWEEKEND', 'ORPROC', 'PAY1']
    )
    nrd2014 = pd.concat([nrd2014, onehot_data_df], axis=1)
    demographic_feature_cols = num_feature_cols + cate_feature_cols
        
    return nrd2014, nrd_icd_cols, nrd_ccs_cols, demographic_feature_cols

def create_label_mimic(admissions_mimic, no_readmit=False):
    """
    no_readmit, True / False: if true, exclude re-admitted hospital records. 
    """
    admissions_mimic['ADMITTIME'] = pd.to_datetime(admissions_mimic['ADMITTIME'])
    admissions_mimic['DISCHTIME'] = pd.to_datetime(admissions_mimic['DISCHTIME'])
    admissions_mimic = admissions_mimic.sort_values(['SUBJECT_ID','ADMITTIME'])
    admissions_mimic['PREV_HADM_TIMES'] = admissions_mimic.groupby('SUBJECT_ID').cumcount()
    admissions_mimic['DISCHTIME_LAST'] = admissions_mimic['DISCHTIME'].shift(1)
    # 30-day readmission label
    admissions_mimic['ADMITTIME_NEXT'] = admissions_mimic['ADMITTIME'].shift(-1)
    admissions_mimic['SUBJECT_ID_NEXT'] = admissions_mimic['SUBJECT_ID'].shift(-1)
    admissions_mimic['GAP_TO_NEXT_ADM'] = (admissions_mimic['ADMITTIME_NEXT'] \
       - admissions_mimic['DISCHTIME']) * (admissions_mimic['SUBJECT_ID'] == \
           admissions_mimic['SUBJECT_ID_NEXT']) + pd.Timedelta(999, unit='D') \
           * (admissions_mimic['SUBJECT_ID'] != admissions_mimic['SUBJECT_ID_NEXT'])
    admissions_mimic['READMISSION'] = (admissions_mimic['GAP_TO_NEXT_ADM'] < pd.Timedelta(30, unit='D')).astype(int)
    # mortality label
    admissions_mimic['DIED'] = admissions_mimic['HOSPITAL_EXPIRE_FLAG']
    # if no_readmit is True, drop all records that ARE ALREADY readmission
    if no_readmit == True:
        admissions_mimic = admissions_mimic[admissions_mimic['PREV_HADM_TIMES'].eq(0)]
    return admissions_mimic

def load_icd9_mimic():
    icd9_mimic0 = pd.read_csv('../data/mimic-iii/DIAGNOSES_ICD.csv')
    icd9_mimic0 = icd9_mimic0[icd9_mimic0.ICD9_CODE.notna()]
    icd9_mimic0['SEQ_NUM'] = icd9_mimic0['SEQ_NUM'].astype(int)
    icd9_mimic = icd9_mimic0.set_index(['HADM_ID','SUBJECT_ID','SEQ_NUM'])\
        .ICD9_CODE.unstack('SEQ_NUM').reset_index()
    icd9_mimic.columns = ['DX'+str(col) if isinstance(col, int) else col for col in icd9_mimic.columns]
    return icd9_mimic

def load_hadm_level_features_mimic(csv_path):
    df = pd.read_csv(csv_path)
    if 'SUBJECT_ID' in df.columns:
        df = df.drop(columns=['SUBJECT_ID'])
    if 'subject_id' in df.columns:
        df = df.drop(columns=['subject_id'])
    assert 'HADM_ID' in df.columns
    feature_cols = [c for c in df.columns if c != 'HADM_ID']
    return df, feature_cols

def process_risk_index_mimic(df, icd_cols, risk_index_path):
    assert 'HADM_ID' in df.columns
    icd_df = df[icd_cols]
    cci = risk_index(icd_df, cci_codes, cci_weights)
    eci = risk_index(icd_df, eci_codes, eci_weights)
    rsi = risk_index(icd_df, rsi_codes, rsi_weights, b=rsi_b)
    risk_index_df = pd.DataFrame()
    risk_index_df['HADM_ID'] = df['HADM_ID']
    risk_index_df['CCI'] = cci
    risk_index_df['ECI'] = eci
    risk_index_df['RSI'] = rsi
    risk_index_df.to_csv(risk_index_path, index=False)
    return risk_index_df

def load_risk_index_mimic(df, icd_cols, disease_icd):
    risk_index_path = '../data/mimic-iii/RISK_INDEX_{}.csv'.format(disease_icd)
    if os.path.exists(risk_index_path):
        risk_index = pd.read_csv(risk_index_path)
    else:
        risk_index = process_risk_index_mimic(df, icd_cols, risk_index_path)
    return risk_index, ['CCI', 'ECI', 'RSI']
    

def process_labevents_mimic(lab_features_path):
    labevents = pd.read_csv('../data/mimic-iii/LABEVENTS.csv')
    d_labitems = pd.read_csv('../data/mimic-iii/D_LABITEMS.csv')
    labevents = labevents.merge(d_labitems, how='left', on='ITEMID')
    labevents = labevents.dropna(subset=['HADM_ID'])
    labevents['ABNORM_FLAG'] = labevents.FLAG.fillna('_').eq('abnormal').astype(float)
    # generate lab value features
    labvalues = labevents.groupby(['HADM_ID', 'LABEL'])['VALUENUM'].agg(['max', 'mean', 'min', 'count']).unstack()
    labvalues.columns = [label+'_'+stat for stat, label in labvalues.columns]
    labvalues = labvalues.dropna(axis=1, thresh=0.5*labvalues.shape[0])
    # generate lab flag features (normal, 0 / abnormal, 1)
    labflags = labevents.groupby(['HADM_ID', 'LABEL'])['ABNORM_FLAG'].agg(['mean', 'sum']).unstack()
    labflags.columns = [label+'_'+stat for stat, label in labflags.columns]
    labflags = labflags.dropna(axis=1, thresh=0.5*labflags.shape[0])
    lab_features = pd.merge(labvalues, labflags, how='outer', on='HADM_ID')
    lab_features.to_csv(lab_features_path)
    return lab_features

def load_lab_features_mimic():
    lab_features_path = '../data/mimic-iii/LAB_FEATURES.csv'
    if os.path.exists(lab_features_path):
        lab_features = pd.read_csv(lab_features_path)
    else:
        lab_features = process_labevents_mimic(lab_features_path)
    
    lab_feature_cols = [c for c in lab_features.columns if c != 'HADM_ID']

    return lab_features, lab_feature_cols

def load_mimic_ts(HADM_ID_df):
    ts_path = '../data/mimic-iii/CHARTEVENTS_ON_GRID.csv'
    mimic_ts = pd.read_csv(ts_path)
    seq_len = mimic_ts['window'].max() - mimic_ts['window'].min() + 1
    
    # some hadm do not have any ts. their 'window' is na
    mimic_ts = pd.merge(HADM_ID_df['HADM_ID'], mimic_ts, how='left', on='HADM_ID')
    mimic_ts_windowisna = mimic_ts[mimic_ts['window'].isna()]
    mimic_ts_windowisna = pd.concat([mimic_ts_windowisna]*seq_len, ignore_index=True)
    mimic_ts_windowisna['window'] = mimic_ts_windowisna.groupby('HADM_ID').cumcount()
    mimic_ts_windowisna = mimic_ts_windowisna.sort_values(['HADM_ID', 'window'])
    mimic_ts = mimic_ts.dropna(subset=['window'])
    mimic_ts = pd.concat([mimic_ts, mimic_ts_windowisna], ignore_index=True)
    mimic_ts = mimic_ts.sort_values(['HADM_ID', 'window'])
    
    # fill na
    mimic_ts = mimic_ts.set_index(['HADM_ID'])
    mimic_ts_mean = mimic_ts.mean(numeric_only=True)
    mimic_ts = mimic_ts.fillna(mimic_ts_mean)
    mimic_ts = mimic_ts.drop(columns=['window'])
    
    # mimic_ts = mimic_ts.groupby('HADM_ID').apply(lambda x: x.values).to_frame()
    # mimic_ts.columns = ['TS']
    mimic_ts = mimic_ts.groupby('HADM_ID').agg('min')
    ts_feature_cols = [c for c in mimic_ts.columns if c != 'HADM_ID' and c!= 'window']
    return mimic_ts, ts_feature_cols

def prepare_mimic_data(disease_icd, no_readmit=True, use_ts_feature=False, 
    use_text_feature=False, use_index_feature = True, use_lab_feature=True, use_demographic=True):
    """
    load mimic iii and create labels
    args:
        disease_icd (str): 'top_15' or '41401', '0389', '41071', '428'
    """
    ## load icd9 diagnosis
    icd9_mimic = load_icd9_mimic()
    if disease_icd.lower() == 'all':
        selected_disease_index = icd9_mimic.DX1.notna()
        icd9_mimic = icd9_mimic[selected_disease_index]   
        print('number of admissions (all):', icd9_mimic.shape[0])
    elif disease_icd.lower() == 'top_15':
        top_15_disease = ('41401', '0389', '41071', '4241', '51881', '431', '486', '5070', '4280', '4240', '430', '5849', '41011', '41041', '5789')
        selected_disease_index = icd9_mimic.DX1.str.startswith(top_15_disease).astype(bool)
        icd9_mimic = icd9_mimic[selected_disease_index]        
        print('number of', top_15_disease,' admissions', icd9_mimic.shape[0])
    else:
        selected_disease_index = icd9_mimic.DX1.str.startswith(str(disease_icd)).astype(bool)
        icd9_mimic = icd9_mimic[selected_disease_index]        
        print('number of', disease_icd,' admissions', icd9_mimic.shape[0])
    ## load admission data / create labels
    admissions_mimic = pd.read_csv('../data/mimic-iii/ADMISSIONS.csv')
    admissions_mimic = create_label_mimic(admissions_mimic, no_readmit)

    ## join icd and labels
    icd9_label_mimic = pd.merge(icd9_mimic, admissions_mimic, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
    icd9_label_mimic.replace('71970', '7197', inplace=True) # 71970 is not in icd9cms, 7197 is
    print('number of all admissions with ICD', icd9_label_mimic.shape)
    print('label ratio', icd9_label_mimic[['READMISSION', 'DIED']].mean())
    mimic_icd_cols = ['DX'+str(i) for i in range(1, 39)]
    
    # merge age and gender
    patients = pd.read_csv('../data/mimic-iii/PATIENTS.csv')
    icd9_label_demo_mimic = pd.merge(icd9_label_mimic, patients, how='left', on=['SUBJECT_ID'])
    icd9_label_demo_mimic['DOB'] = pd.to_datetime(icd9_label_demo_mimic['DOB'])    
    icd9_label_demo_mimic['AGE'] = icd9_label_demo_mimic.apply(
        lambda e: (e['ADMITTIME'].date() - e['DOB'].date()).days // 365, axis=1)
    icd9_label_demo_mimic['AGE'] = icd9_label_demo_mimic.apply(
        lambda e: e['AGE'] if e['AGE'] <=89 else 89, axis=1) # elder people are recorded as 300+ years old
    num_feature_cols = ['AGE', 'PREV_HADM_TIMES']
    # merge time series feature
    if use_ts_feature == True:
        mimic_ts, ts_feature_cols = load_mimic_ts(icd9_label_demo_mimic)
        icd9_label_demo_mimic = icd9_label_demo_mimic.merge(mimic_ts, on='HADM_ID', how='left')
        num_feature_cols += ts_feature_cols

    # merge text feature
    if use_text_feature == True:
        texts = get_mimic_notes(icd9_label_demo_mimic, disease_icd)
        icd9_label_demo_mimic = icd9_label_demo_mimic.merge(texts, on='HADM_ID', how='left')

    # merge lab features
    if use_lab_feature == True:
        lab_features, lab_feature_cols = load_lab_features_mimic()
        icd9_label_demo_mimic = icd9_label_demo_mimic.merge(lab_features, on='HADM_ID', how='left')
        num_feature_cols += lab_feature_cols
        
    # merge risk index as features
    if use_index_feature == True:
        risk_index, risk_index_cols = load_risk_index_mimic(icd9_label_demo_mimic, mimic_icd_cols, disease_icd)
        icd9_label_demo_mimic = icd9_label_demo_mimic.merge(risk_index, on='HADM_ID', how='left')
        num_feature_cols += risk_index_cols

    if use_demographic == True:
        # these are demographic features
        onehot_data_df, cate_feature_cols = one_hot_with_column_name(
            icd9_label_demo_mimic, 
            ['INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'GENDER']
        )
        icd9_label_demo_mimic = pd.concat([icd9_label_demo_mimic, onehot_data_df], axis=1)
        tabular_feature_cols = num_feature_cols + cate_feature_cols
    else:
        tabular_feature_cols = num_feature_cols

    return icd9_label_demo_mimic, mimic_icd_cols, tabular_feature_cols
    
def get_mimic_notes(HADM_ID_df, disease_icd):
    note_path = '../data/mimic-iii/NOTEEVENTS_{}.csv'.format(disease_icd)
    if os.path.exists(note_path):
        noteevents = pd.read_csv(note_path)
    else:
        noteevents = pd.read_csv('../data/mimic-iii/NOTEEVENTS.csv')
        noteevents = noteevents[noteevents['CATEGORY'] == 'Discharge summary']
        noteevents = pd.merge(HADM_ID_df['HADM_ID'], noteevents, how='left', on='HADM_ID')
        noteevents.to_csv(note_path, index=False)
    
    noteevents['TEXT'] = noteevents['TEXT'].astype(str)
    noteevents['TEXT'] = noteevents['TEXT'].apply(remove_sentence_has_keywords)
    noteevents['TEXT'] = noteevents['TEXT'].astype(str)
    hosp_noteevents = noteevents.groupby(['HADM_ID'])['TEXT'].agg(lambda x: '\n'.join(x)).reset_index()
    return hosp_noteevents

def remove_sentence_has_keywords(str_to_clean):
    keywords = ['die ', 'death ', 'expire ', 'dies ', 'died ', 'expired ', 'expires ']
    pattern = '|'.join(keywords)
    cleaned_str = ""
    for sentence in re.split('(\.|\n)', str_to_clean):
        if not (re.search(pattern, sentence, flags=re.IGNORECASE)):
            cleaned_str = cleaned_str + sentence
    return cleaned_str
    

def one_hot_with_column_name(df, cols):
    df = df.fillna('-99')
    encoder = OneHotEncoder(sparse = False)
    onehot_data_arr = encoder.fit_transform(df[cols])    
    categories = encoder.categories_
    feature_names = ['{}_{}'.format(c, each_cat) for c, category_arr in zip(cols, categories) for each_cat in category_arr]
    onehot_data_df = pd.DataFrame(onehot_data_arr, columns = feature_names)
    return onehot_data_df, feature_names
    
def get_data_for_task(data, task_name):
    assert task_name.lower() in ['readmission', 'mortality']
    if task_name == 'readmission':
        task_data = data[data['DIED']==0].reset_index()
    else:
        task_data = data
    return task_data

def binarize(med_code_df):
    """
    Binarize medical codes for each visit

    Parameters
    ----------
    med_code_df : pandas df
        a dataframe of medical codes, with each row as a visit

    Returns
    -------
    An array of 0/1 that can be used for training/testing

    """
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(med_code_df.fillna('-99').values)
    X = X[:, mlb.classes_ != '-99']
    return X
    
def transform_code_to_index(data, code_list):
    code2index = dict(zip(code_list, list(range(len(code_list)))))
    indexed_data = data.apply(lambda each_code_set: {code2index[c] for c in each_code_set})
    return indexed_data
def transform_code_df_to_set_series(code_df):
    code_set = code_df.apply(lambda x: set(x), axis=1)
    return code_set

def cbind_df(df1, df2):
    combined_df = pd.concat((df1, df2), axis=1, ignore_index=True)
    return combined_df