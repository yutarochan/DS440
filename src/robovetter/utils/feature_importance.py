'''
Feature Importance Analysis Script
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import time
import random
import numpy as np
import pandas as pd
import datetime as dt
import gc; gc.enable()
from astropy.io import ascii

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomTreesEmbedding

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Initialize Timer
t0 = time.time()

# Initialize PRNG Seed
random.seed(9892)
np.random.seed(9892)

# Load Dataset
print ("-" * 100)
print ("Load Training File")
data = ascii.read('data/raw/tces/kplr_dr25_inj1_tces.txt').to_pandas()

# Augment Target Value Encoding
data['Disp'].replace('FP', 0, inplace=True)
data['Disp'].replace('PC', 1, inplace=True)

# Separate Features and Target Values
print ("-" * 100)
print ("Set X Features and y Target")
feat = ['Score', 'NTL', 'SS', 'CO', 'EM', 'period', 'epoch', 'Expected_MES', 'MES',
        'NTran', 'depth', 'duration', 'Rp', 'Rs', 'Ts', 'logg', 'a', 'Rp/Rs', 'a/Rs',
        'impact', 'SNR_DV', 'Sp', 'Fit_Prov']
y = data['Disp'].values

data.drop(columns=['Disp'])
X = data[feat]
names = X.columns

print(X.head())

# Perform Data Imputation and PCA Computations
print ("-" * 100)
print ("Perform Imputation and Scaling for PCA Calculations")
for c in X.columns:
    if X[c].dtype=='object':
        X[c] = X[c].fillna(-999)
        X[c] = pd.factorize(X[c], sort=True)[0]
    else:
        X[c] = X[c].astype('float64')
        X[c] = X[c].fillna(-999)
        rscaler = StandardScaler()
        rscaler.fit(X[c].values.reshape(-1,1))
        X[c] = rscaler.transform(X[c].values.reshape(-1,1))

# Clean Up GC
del data; gc.collect()

# Compute PCA Components
'''
print ("-" * 100)
print ("Compute for the PCA Components and Add into Data Set")
# PCA --> 85% = 13 | 90% = 26 | 95% = 57
pca = PCA(n_components=X.shape[1])
X_pca = pd.DataFrame(pca.fit_transform(X))
X_pca.columns = ['pca'+str(i) for i in range(1, X.shape[1]+1)]
names = names.append(X_pca.columns)
X_all = pd.concat([X, X_pca], axis=1)
'''
X_all = X

# Compute Various Feature Importance Metrics
print ("-" * 100)
print ("Compute Using Scikit FI Algorithms")
for n_algo in ['rfr','ada','ext','gbm','rte']:
    # Choose and set the algorithm to run here
    #    Random Forest          = 'rfr'
    #    AdaBoost               = 'ada'
    #    Extra Trees            = 'ext'
    #    Gradient Boosting      = 'gbm'
    #    Random Trees Embedding = 'rte'
    print ('   --> Algorithm:', n_algo)
    if   n_algo == 'rfr':
        fi_model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=0)
    elif n_algo == 'ada':
        fi_model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.50, random_state=0)
    elif n_algo == 'ext':
        fi_model = ExtraTreesClassifier(n_estimators=1000, max_features=0.90, random_state=0)
    elif n_algo == 'gbm':
        fi_model = GradientBoostingClassifier(n_estimators=1000, max_features=0.90, max_depth=6, random_state=0)
    elif n_algo == 'rte':
        fi_model = RandomTreesEmbedding(n_estimators=1000, random_state=0)

    print ('       Train:', n_algo)
    fi_model.fit(X_all, y)
    print ('       Save:', n_algo)
    fi = sorted(zip(map(lambda x: round(x, 4), fi_model.feature_importances_), names), reverse=True)
    df_fi = pd.DataFrame.from_records(fi)
    df_fi.to_csv('data/feat_eng/rv/_' + n_algo + '_feature_importance.csv', sep=',')

print ("-" * 100)
print ("Perform Chi^2 Feature Selection and Save Results to File")
sel = SelectKBest(chi2, k='all') # (All) X features (you can use reduced k as a cutoff also)
train_x2 = X_all**2 # use square of X as chi2 can not work with negative values
sel.fit(train_x2, y)
sel.transform(train_x2)
df_univ = pd.DataFrame()
df_univ["feature"] = [x for x in names]
df_univ["score"] = [x**(1/2) for x in sel.scores_]
df_univ.to_csv("data/feat_eng/rv/_uni_feature_importance.csv", index=False)

print ("-" * 100)
print ("Perform Lasso Feature Selection and Save Results to File")
print ("   --> Before Lasso Features: %i" % X_all.shape[1])
sel = SelectFromModel(LassoCV(n_alphas=200, alphas=[0.01], max_iter=2000, cv=5, random_state=0, n_jobs=-1))
sel.fit(X_all, y)
lasso_sel = sel.transform(X_all)
print ("   --> After Lasso Selection: %i" % lasso_sel.shape[1])
df_lasso = pd.DataFrame()
df_lasso["feature"] = [x for x in names]
df_lasso["include"] = [1 if x==True else 0 for x in sel.get_support()]
df_lasso.to_csv("data/feat_eng/rv/_las_feature_selection.csv", index=False)

print ("-" * 100)
t1 = time.time()
print ("Done with Process in %.2f mins" % ((t1-t0)/60))
