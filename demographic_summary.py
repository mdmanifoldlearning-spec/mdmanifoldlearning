# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:54:10 2021

@author: Shaodong
"""
import pandas as pd
from data_processing_tools import (one_hot_with_column_name)
import matplotlib.pyplot as plt

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
print('AGE', nrd2014.AGE.astype(int).min(), nrd2014.AGE.astype(int).mean(), nrd2014.AGE.astype(int).max())
print('Male', nrd2014.FEMALE.astype(int).count() - nrd2014.FEMALE.astype(int).sum())
print('Female', nrd2014.FEMALE.astype(int).sum())
print(nrd2014.PAY1.value_counts())

#%%
"""
load mimic iii and create labels
"""
disease_icd = '41401'
## load icd9 diagnosis
icd9_mimic0 = pd.read_csv('../data/mimic-iii/DIAGNOSES_ICD.csv')
icd9_mimic0 = icd9_mimic0[icd9_mimic0.ICD9_CODE.notna()]
icd9_mimic0['SEQ_NUM'] = icd9_mimic0['SEQ_NUM'].astype(int)
icd9_mimic = icd9_mimic0.set_index(['HADM_ID','SUBJECT_ID','SEQ_NUM'])\
    .ICD9_CODE.unstack('SEQ_NUM').reset_index()
icd9_mimic.columns = ['DX'+str(col) if isinstance(col, int) else col for col in icd9_mimic.columns]
print('Max number of ICD in each admission', sum([1 for col in icd9_mimic.columns if 'DX' in col]))
## load admission data / create labels
admissions_mimic = pd.read_csv('../data/mimic-iii/ADMISSIONS.csv')
print('number of all admissions', admissions_mimic.shape[0])
## DX1 counts
from icd9cms.icd9 import search
DX1_counts = icd9_mimic.DX1.value_counts().to_frame()
DX1_counts.columns = ['DX1_counts']
DX1_counts['Desc'] = [search(icd).__str__() for icd in DX1_counts.index]
plt.figure(figsize=(16, 12), dpi=80)
plt.plot(DX1_counts.head(50).index, DX1_counts.head(50).DX1_counts)
plt.xticks(rotation=45, ha='right')
plt.show()

if disease_icd.lower() != 'all':
    selected_disease_index = icd9_mimic.DX1.str.startswith(str(disease_icd)).astype(bool)
    icd9_mimic = icd9_mimic[selected_disease_index]        
    print('number of', disease_icd,' admissions', icd9_mimic.shape[0])
admissions_mimic['ADMITTIME'] = pd.to_datetime(admissions_mimic['ADMITTIME'])
admissions_mimic['DISCHTIME'] = pd.to_datetime(admissions_mimic['DISCHTIME'])
admissions_mimic = admissions_mimic.sort_values(['SUBJECT_ID','ADMITTIME'])
admissions_mimic['ADMITTIME_NEXT'] = admissions_mimic['ADMITTIME'].shift(-1)
admissions_mimic['SUBJECT_ID_NEXT'] = admissions_mimic['SUBJECT_ID'].shift(-1)
admissions_mimic['GAP'] = (admissions_mimic['ADMITTIME_NEXT'] \
   - admissions_mimic['DISCHTIME']) * (admissions_mimic['SUBJECT_ID'] == \
       admissions_mimic['SUBJECT_ID_NEXT']) + pd.Timedelta(999, unit='D') \
       * (admissions_mimic['SUBJECT_ID'] != admissions_mimic['SUBJECT_ID_NEXT'])
admissions_mimic['READMISSION'] = (admissions_mimic['GAP'] < pd.Timedelta(30, unit='D')).astype(int)
admissions_mimic['DIED'] = admissions_mimic['HOSPITAL_EXPIRE_FLAG']
## join icd and labels
icd9_label_mimic = pd.merge(icd9_mimic, admissions_mimic, how='inner', on=['HADM_ID', 'SUBJECT_ID'])
icd9_label_mimic.replace('71970', '7197', inplace=True) # 71970 is not in icd9cms, 7197 is
print('number of all admissions with ICD', icd9_label_mimic.shape)
print('label ratio', icd9_label_mimic[['READMISSION', 'DIED']].mean())
mimic_icd_cols = ['DX'+str(i) for i in range(1, 39)]

## merge demographic info
patients = pd.read_csv('../data/mimic-iii/PATIENTS.csv')
icd9_label_demo_mimic = pd.merge(icd9_label_mimic, patients, how='left', on=['SUBJECT_ID'])
icd9_label_demo_mimic['DOB'] = pd.to_datetime(icd9_label_demo_mimic['DOB'])    
icd9_label_demo_mimic['AGE'] = icd9_label_demo_mimic.apply(
    lambda e: (e['ADMITTIME'].date() - e['DOB'].date()).days // 365, axis=1)
icd9_label_demo_mimic['AGE'] = icd9_label_demo_mimic.apply(
    lambda e: e['AGE'] if e['AGE'] <=89 else 89, axis=1) # elder people are recorded as 300+ years old

print('AGE\n', icd9_label_demo_mimic.AGE.mean())
print('GENDER\n', icd9_label_demo_mimic.GENDER.value_counts())
print('Ethnicity\n', icd9_label_demo_mimic.ETHNICITY.value_counts())
print('Insurance\n', icd9_label_demo_mimic.INSURANCE.value_counts())
print('MARITAL STATUS\n', icd9_label_demo_mimic.MARITAL_STATUS.value_counts())

# %%
