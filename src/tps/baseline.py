'''
TPS Baseline Model
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

from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

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

# Binarize Labels
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y = lb.transform(y)

# Define Model Function
def tps_gamma(exp_mes):
    return 0.94 * stats.gamma.cdf(exp_mes, 30.87, scale=0.271)

print ("-" * 100)
print ("Perform Cross Validation Test")
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    x_train = np.take(X.values, train_index)
    y_train = np.take(y, train_index)

    pred = tps_gamma(x_train)
    y_hat = [0 if p <= 0.5 else 1 for p in pred]
    print(classification_report(y_train, y_hat))
