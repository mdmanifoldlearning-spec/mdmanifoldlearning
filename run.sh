#!/usr/bin/env bash

# 414.01 (‘Coronary atherosclerosis of native coronary artery’), accounting for 7.1% of all hospital admissions;

# 038.9 (‘Unspecified septicemia’), accounting for 4.2% of all hospital admissions; and

# 410.71 (‘Subendocardial infarction, initial episode of care’), accounting for 3.6% of all hospital admissions.

python eval_baselines.py -d mimic -s top_15 -e True
python eval_baselines.py -d mimic -s top_15 -e False
# python generate_distance.py -d mimic -s 0389 -t readmission -i Cosine -g None -v SS7 -n 5
# python dimension_reduction.py -d mimic -s 0389 -t readmission
# python prediction.py -d mimic -s 0389 -t readmission

# for selected_disease in 0389 41071 41401; do
#     for task_name in readmission mortality; do
#         for visit_method in SS5 SS6 SS7 SS8; do
#         python generate_distance.py -d mimic -s $selected_disease -t $task_name -i Cosine -g None -v $visit_method -n 5
#         python generate_distance.py -d mimic -s $selected_disease -t $task_name -i Group_plus_Cosine -g ccs -v $visit_method -n 5
#         python generate_distance.py -d mimic -s $selected_disease -t $task_name -i Group_plus_Cosine -g cui -v $visit_method -n 5
#         done
#     done
# done