'''
Feature Generation - Script for Feature Generation
Author: Yuya Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import itertools
import numpy as np
import pandas as pd
from astropy.io import ascii

# Center Pool Mapping
panel = 2
center_map = {}
for i in range(1, 85):
    center_map[i] = panel
    if i % 4 == 0:
        if panel == 4 or panel == 20:
            panel += 2
        else:
            panel += 1

# Manually Define Index Pairs for All Combinations
single = [4, 11, 15, 32, 56, 71, 75, 84]
double = [(3,8), (7,12), (16,35), (31,52), (36,55), (51,72), (76,79), (80,83)]
triple = [(1,14,20), (10,28,29), (53,60,74), (68,70,81)]
quad = [(2,5,19,24), (6,9,23,25), (13,17,34,39), (18,21,38,43), (22,26,42,45), (27,30,48,49), (33,40,54,59),
        (37,44,58,62), (41,46,61,66), (47,50,65,69), (57,63,73,78), (64,67,77,82)]

# Joint Type Mapping
join_type = {}
for i in single: join_type[i] = 1
for i in list(itertools.chain(*double)): join_type[i] = 2
for i in list(itertools.chain(*triple)): join_type[i] = 3
for i in list(itertools.chain(*quad)): join_type[i] = 4

# Corner Group Mapping
corner_map = {}
for i, x in enumerate(single + double + triple + quad):
    if isinstance(x, int):
        corner_map[x] = i
    else:
        for j in x: corner_map[j] = i

t_1 = [2, 3, 4, 7, 8]
t_2 = [6, 11, 12, 13, 16, 17]
t_3 = [9, 10, 14, 15, 20]
t_4 = [18, 19, 22, 23, 24]

ch_ori = {}
for t in t_1: ch_ori[t] = 1
for t in t_2: ch_ori[t] = 2
for t in t_3: ch_ori[t] = 3
for t in t_4: ch_ori[t] = 4

# Load Dataset
data = ascii.read('data/raw/plti/kplr_dr25_inj1_plti.txt').to_pandas()

# Process Features
data['sg_center_pool'] = data['Sky_Group'].apply(lambda x: center_map[x])
data['sg_corner_pool'] = data['Sky_Group'].apply(lambda x: corner_map[x])
data['sg_ori_pool'] = data['Sky_Group'].apply(lambda x: ch_ori[center_map[x]])

# Export Data
data.to_csv('data/raw/plti/kplr_dr25_inj1_plti_sgmod.csv', header=True, index=False)
