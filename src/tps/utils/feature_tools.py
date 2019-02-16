'''
FeatureTools: Automated Feature Generator
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import time
import numpy as np
import pandas as pd
import gc; gc.enable()
import featuretools as ft
from astropy.io import ascii

import featuretools.primitives
from featuretools.primitives import AddNumeric

# Initialize Timer
t0 = time.time()

# Load Dataset
print ("-" * 100)
print ("Load Training File")
data = ascii.read('data/raw/plti/kplr_dr25_inj1_plti.txt').to_pandas()

print(data.head())

# Separate Features and Target Values
print ("-" * 100)
print ("Set X Features and y Target")
feat = ['KIC_ID', 'Sky_Group', 'i_period', 'i_epoch', 'N_Transit', 'i_depth', 'i_dur', 'i_b', 'i_ror', 'i_dor', 'Expected_MES']
data = data[feat]

# Build Entity Set and Append Data Entity
es = ft.EntitySet(id = 'kplr')
es.entity_from_dataframe(entity_id = 'data', dataframe = data, index='KIC_ID')

# Generate Feature Matrix
prims = ['add_numeric', 'subtract_numeric', 'multiply_numeric', 'divide_numeric', 'modulo_by_feature', 'divide_by_feature']
for p in prims:
    feature_matrix, feature_names = ft.dfs(entityset=es, target_entity = 'data',
        max_depth = 2, verbose = 1, n_jobs = 1,
        trans_primitives = [p])
    feature_matrix.to_csv('data/feat_eng/plti_' + p + '.csv')
