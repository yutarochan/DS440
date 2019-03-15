'''
Feature Cluster Analysis
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import ascii
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load OG Dataset
print ("-" * 100)
print ('Load Original Data Set')
df = ascii.read('data/raw/plti/kplr_dr25_inj1_plti.txt').to_pandas()
print ("   ", df.shape)

# Drop Unique Identifiers
print ("-" * 100)
print ('Select Target Features')
feat = ['Sky_Group', 'i_period', 'i_epoch', 'N_Transit', 'i_depth', 'i_dur', 'i_b', 'i_ror', 'i_dor', 'Expected_MES']
df = df[feat]
print ("   ", df.shape)
