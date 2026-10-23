# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:26:32 2021

@author: Shaodong
"""


from os import path
from numpy.ma.core import where
# from code9cms.code9 import search
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import diags
import time
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
import gc
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances, euclidean_distances
from utils import dump_pickle, load_pickle


class ConceptLevelDistance:
    
    def __init__(self, all_codes, codeidx2ancesidx, hierarchy_name = 'icd'):
        """
        all_codes : TYPE
            list of medical codes. It can be a combination of icd and 
            ccs/cui/snomed.
        codeidx2ancesidx : dict (int -> list of int)
            a dict from the index of code to the indices of its ancestors.
            The ancestors in the list are ordered from itself (not included) 
            to the root.
            The code itself is not included in the list. 
            e.g. {1: [2,3], 2:[3]}
        hierarchy_name : TYPE, optional
            one of icd, icd_ccs, icd_cui, icd_snomed. The default is 'icd'.
        """
        self.all_codes = all_codes
        self.n_codes = len(all_codes)
        self.code2idx = dict(zip(self.all_codes,range(self.n_codes)))
        self.idx2code = dict(zip(range(self.n_codes), self.all_codes))
        self.codeidx2ancesidx = codeidx2ancesidx
        self.hierarchy_name = hierarchy_name
        
                
    def get_all_codes_dist(
                        self,
                        code_idx_sets = None,
                        code_method="IC1CS2",
                        code_loc=None,
                        code_recalc=True):
        assert code_method in [
            'IC1CS2', 
            'Cosine', 'Group_plus_Cosine',
            "Euclidean", "Group_plus_Euclidean",
            "Manhattan", "Group_plus_Manhattan",
            "eHDN", "Group_plus_eHDN"
        ]
        
        saved_path = code_loc+code_method+ '_'+ self.hierarchy_name+'.pickle'
        if code_recalc == False or path.exists(saved_path):
            (code_dist, code2idx, idx2code) = load_pickle(saved_path) 
            return code_dist, code2idx, idx2code
        
        if code_method == "IC1CS2":
            code_dist = [
                [self.get_a_IC1CS2(code1idx, code2idx) for code2idx in range(code1idx)]+[0]*(self.n_codes-code1idx)
                for code1idx in range(self.n_codes)
            ]
            
            code_dist = np.array(code_dist,dtype=np.float32)
            code_dist = code_dist + code_dist.transpose()

        assert code_idx_sets is not None
        if code_method in ["Cosine", "Group_plus_Cosine"]:
            W = self.get_cooccurrence_matrix(code_idx_sets = code_idx_sets)
            code_dist = cosine_distances(W)
        if code_method in ["Euclidean", "Group_plus_Euclidean"]:
            W = self.get_cooccurrence_matrix(code_idx_sets = code_idx_sets)
            code_dist = euclidean_distances(W)
        if code_method in ["Manhattan", "Group_plus_Manhattan"]:
            W = self.get_cooccurrence_matrix(code_idx_sets = code_idx_sets)
            code_dist = manhattan_distances(W)

        if code_method in ["eHDN", "Group_plus_eHDN"]:
            """
            https://doi.org/10.1038/s41598-018-21779-y
            """
            X = self.get_occurrence_matrix(code_idx_sets = code_idx_sets)
            P = X.A.sum(axis=0)
            C = self.get_cooccurrence_matrix(code_idx_sets = code_idx_sets)
            N = code_idx_sets.shape[0]

            numerator = N*C - np.outer(P,P)
            denominator = np.sqrt(np.outer(P,P)) * np.sqrt(np.outer(N-P, N-P))
            # denominator may contain 0 because some disease may not appear in data
            S = np.divide(numerator, denominator, out = np.zeros_like(denominator), where=denominator!=0)
            code_dist = 1- S

        if code_loc != None:
            dump_pickle(saved_path, (code_dist,self.code2idx,self.idx2code))
        return  code_dist, self.code2idx, self.idx2code
    
    def get_a_IC1CS2(self, code1idx, code2idx):
        """
        get a distance for a code pair based on IC1 and CS2
        IC of a virtual root is 1; IC of a chapter is 2.
        args:
            code1idx (int): index of first code code
            code2idx (int): index of second code code
        returns:
            dis (float): distance 
        """
        ances1 = self.codeidx2ancesidx[code1idx]
        ances2 = self.codeidx2ancesidx[code2idx]
        IC1 = len(ances1)+1
        IC2 = len(ances2)+1
        IC3 = IC1
        for a in ances1: 
            if a not in ances2:
                IC3 = IC3 - 1
            else:
                break
        dis = 1-2*IC3/(IC1+IC2)
        return dis
    
    
    def get_cooccurrence_matrix(self, code_idx_sets):
        X = self.get_occurrence_matrix(code_idx_sets)
        W = X.transpose().dot(X)
        return W
    
    def get_occurrence_matrix(self, code_idx_sets):
        """
        return a occurrence matrix (sparse, csr format) 
        with the shape of (#visits, #medical codes).
        """
        ances_idx_sets = code_idx_sets.apply(
            lambda x:
            {e for r in x for e in self.codeidx2ancesidx[r]}.union(x)
        )
            
        i_j_idx = [(i,j) for i in range(ances_idx_sets.shape[0]) \
                     for j in ances_idx_sets[i]]
        i_j_idx = np.array(i_j_idx)
        i_idx = i_j_idx[:,0]
        j_idx = i_j_idx[:,1]
        del i_j_idx
        X = csr_matrix((np.ones_like(i_idx),(i_idx,j_idx)),\
                       shape=(ances_idx_sets.shape[0],self.n_codes))
        return X

class SetLevelDistance:
    @staticmethod
    def get_a_SS5(idx_set1, idx_set2, code_dist):
        weight = code_dist[list(idx_set1),:][:,list(idx_set2)]
        part1 = weight.min(axis=0).sum()
        part2 = weight.min(axis=1).sum()
        dis = (part1 + part2) / (len(idx_set1) + len(idx_set2))
        return dis
    
    @staticmethod
    def get_a_SS6(idx_set1, idx_set2, code_dist):
        A_B = idx_set1 - idx_set2 # set1\set2
        B_A = idx_set2 - idx_set1 # set2\set1
        part1 = code_dist[list(A_B),:][:,list(idx_set2)].sum()/len(idx_set2)
        part2 = code_dist[list(idx_set1),:][:,list(B_A)].sum()/len(idx_set1)
        dis = (part1 + part2) / len(idx_set1.union(idx_set2))
        return dis
    
    @staticmethod
    def get_a_SS7(idx_set1, idx_set2, code_dist):
        i,j = np.meshgrid(list(idx_set1), list(idx_set2), indexing='ij')
#        i,j = np.meshgrid(idx_set1, idx_set2, indexing='ij')
        dis = code_dist[i.flatten(),j.flatten()]
        dis = sum(dis) / len(dis)
        return dis
    
    @staticmethod
    def get_a_SS8(idx_set1,idx_set2, code_dist):
        weight = code_dist[list(idx_set1),:][:,list(idx_set2)]
        # minimal weighted bipartite matching
        dis = weight[linear_sum_assignment(weight)].mean()
        return dis

def visit_dist_helper(dist_func, from_idx, to_idx, code_dist, code_idx_sets):
    n_visits = len(code_idx_sets)
    def cal_a_row(dist_func, i):
        if i % 1000 == 0:
            print(i)
        a_row_in_dis_matrix = [
            dist_func(code_idx_sets[i], idx_set2, code_dist) 
            for idx_set2 in code_idx_sets[0:i]
        ] + [0]*(n_visits-i)
        return a_row_in_dis_matrix
    return [cal_a_row(dist_func, i) for i in range(from_idx, to_idx)]

def get_all_visit_dist(code_idx_sets, code_dist, concept_level_method, hierarchy_name, set_level_method, n_jobs, visit_loc=None):
    if visit_loc is not None:
        path_to_save = visit_loc + concept_level_method + '_' + hierarchy_name + '_'+ set_level_method + '.pickle'
        if path.exists(path_to_save):
            dis_matrix = load_pickle(path_to_save) 
            return dis_matrix

    n_visits = len(code_idx_sets)
    
    assert set_level_method in ['SS5', 'SS6', 'SS7', 'SS8']
    if set_level_method == 'SS5':
        get_distance_between_two_visits = SetLevelDistance.get_a_SS5
    elif set_level_method == 'SS6':
        get_distance_between_two_visits = SetLevelDistance.get_a_SS6
    elif set_level_method == 'SS7':
        get_distance_between_two_visits = SetLevelDistance.get_a_SS7
    elif set_level_method == 'SS8':
        get_distance_between_two_visits = SetLevelDistance.get_a_SS8
    
    start =time.time()
    if n_jobs == 1:
        dis_lower_triangle = visit_dist_helper(get_distance_between_two_visits, 0, n_visits, code_dist, code_idx_sets)
        # [
        #     visit_dist_helper(get_distance_between_two_visits, i, code_dist, code_idx_sets)
        #     for i in range(n_visits)
        # ]
    else: 
        mp_job_partition_start_idx = [int(np.sqrt(i/n_jobs)*n_visits) for i in range(n_jobs)]
        mp_job_partition_end_idx = [int(np.sqrt(i/n_jobs)*n_visits) for i in range(1, n_jobs)]+[n_visits]
        par_args = zip(
            [get_distance_between_two_visits]*n_jobs, 
            mp_job_partition_start_idx,
            mp_job_partition_end_idx,
            [code_dist]*n_jobs,
            [code_idx_sets]*n_jobs
        )
        with mp.Pool(n_jobs) as pool:
            multiprocess_out = pool.starmap(visit_dist_helper, par_args)
            dis_lower_triangle = [each_row for dist_block in multiprocess_out for each_row in dist_block]
    dis_lower_triangle_arr = np.array(dis_lower_triangle, dtype=np.float32)
    dis_matrix = dis_lower_triangle_arr + dis_lower_triangle_arr.transpose()
    print('Time used (s)', time.time()-start)
    if visit_loc is not None:
        path_to_save = visit_loc + concept_level_method + '_' + hierarchy_name + '_'+ set_level_method + '.pickle'
        dump_pickle(path_to_save, dis_matrix)
    return dis_matrix
    
    
    
if __name__ == '__main__':
    N = 1000
    all_codes = list('abcdefghijklmnopqrstuvwxyz')
    code2idx = dict(zip(all_codes, range(len(all_codes))))
    # code2ances = {e:['z'] for e in all_codes}
    codeidx2ancesidx = {code2idx[e]:[code2idx['z']] for e in all_codes}
    code_idx_sets = pd.DataFrame(np.random.randint(0,len(all_codes), size=(N,30))).apply(lambda x: set(x), axis=1)
    dis = ConceptLevelDistance(
            all_codes = all_codes,
            codeidx2ancesidx = codeidx2ancesidx,
            hierarchy_name = 'intrinsic_icd'
        )
    code_dist,_,_ = dis.get_all_codes_dist(code_idx_sets= code_idx_sets, 
                                    code_method="IC1CS2",
                                    code_loc='../data/test/', 
                                    code_recalc=True)
    code_dist,_,_ = dis.get_all_codes_dist(code_idx_sets= code_idx_sets, 
                                    code_method="Cosine",
                                    code_loc='../data/test/', 
                                    code_recalc=True)

    visit_dist = get_all_visit_dist(code_idx_sets, 
                       code_dist, 
                       concept_level_method = 'Cosine',
                       set_level_method = 'SS5', 
                       n_jobs = 5, 
                       hierarchy_name = 'intrinsic_icd',
                       visit_loc='../data/test/')
    
    
    
    
    
    
    