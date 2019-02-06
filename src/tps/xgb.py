'''
TPS XGBoost Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import gc
import math
import time
import random
import numpy as np
import pandas as pd

from utils.data import DataLoader

import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Initialize Pipeline
t0 = time.time()

# Initialize PRNG Seed
random.seed(9892)
np.random.seed(9892)

# Load Dataset
print ("-" * 100)
print ("Load Training File")
dl = DataLoader('data/raw/plti/kplr_dr25_inj1_plti.txt')
X, y = dl.load_data()

# Data Imputation Process
print ("-" * 100)
print ("Impute, Factorize, and Scale Features")
for c in X.columns:
    if X[c].dtype == 'object':
        X[c] = X[c].fillna(-1)
        X[c] = pd.factorize(X[c], sort=True)[0]
    if X[c].dtype == np.float64:
        X[c] = X[c].astype(np.float32)
        X[c] = X[c].fillna(-999)
        rscaler = StandardScaler()
        X[c] = rscaler.fit_transform(X[c].values.reshape(-1,1))

print ("-" * 100)
print ("Set XGBoost Parameters")
xgb_params = {
    'eta': 0.01,                    # 0.01
    'max_depth': 18,                 # 6
    'min_child_weight': 1,
    'subsample': 0.80,              # 0.80
    'objective': 'binary:logistic',
    'colsample_bytree': 0.50,
    'scale_pos_weight': 2,
    'eval_metric': 'auc',
    'base_score': np.mean(y),
    'gpu_id': 0,
    'seed': 9389493,
    'silent': 1
}

print ("-" * 100)
print ("Perform Cross Validation Test")
dtrain = xgb.DMatrix(X.values, y.values)
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold=10,                    # 10
                   num_boost_round=6000,        # set to 5000 only for testing
                   early_stopping_rounds=50,
                   verbose_eval=1,
                   show_stdv=False
                  )

print(cv_result)

num_boost_rounds = len(cv_result)
print ("Optimal No. of Rounds: %i" % (num_boost_rounds))
