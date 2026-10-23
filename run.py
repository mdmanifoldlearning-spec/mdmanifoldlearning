# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:01:37 2021

414.01 (‘Coronary atherosclerosis of native coronary artery’), accounting for 7.1% of all hospital admissions;

038.9 (‘Unspecified septicemia’), accounting for 4.2% of all hospital admissions; and

410.71 (‘Subendocardial infarction, initial episode of care’), accounting for 3.6% of all hospital admissions.

@author: Shaodong
"""

 

!python generate_distance.py -d mimic -s 41401 -t readmission -i Cosine -g None -v SS5 -n 3
!python generate_distance.py -d mimic -s 41401 -t readmission -i Cosine -g None -v SS6 -n 3
!python generate_distance.py -d mimic -s 41401 -t readmission -i Cosine -g None -v SS7 -n 3
!python generate_distance.py -d mimic -s 41401 -t readmission -i Cosine -g None -v SS8 -n 3

!python generate_distance.py -d mimic -s 41401 -i Group_plus_Cosine -g ccs -v SS5 -n 3
!python generate_distance.py -d mimic -s 41401 -i Group_plus_Cosine -g ccs -v SS6 -n 3
!python generate_distance.py -d mimic -s 41401 -i Group_plus_Cosine -g ccs -v SS7 -n 3
!python generate_distance.py -d mimic -s 41401 -i Group_plus_Cosine -g ccs -v SS8 -n 3

!python generate_distance.py -d mimic -s "sepsis" -i Group_plus_Cosine -g cui -v SS5 -n 3
!python generate_distance.py -d mimic -s "sepsis" -i Group_plus_Cosine -g cui -v SS6 -n 3
!python generate_distance.py -d mimic -s "sepsis" -i Group_plus_Cosine -g cui -v SS7 -n 3
!python generate_distance.py -d mimic -s "sepsis" -i Group_plus_Cosine -g cui -v SS8 -n 3

!python dimension_reduction.py -d mimic -s "sepsis"
!python dimension_reduction.py -d mimic -s "heart failure"

!python prediction.py -d mimic -s "sepsis"
!python prediction.py -d mimic -s "heart failure"

!python eval_baselines.py -d mimic -s 41071
!python eval_baselines.py -d mimic -s 41401