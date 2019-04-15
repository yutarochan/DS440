'''
TPS Pipeline Efficiency Dataset
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import math
import random
import pprint
import numpy as np
import pandas as pd
from astropy.io import ascii

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, config):
        print('='*80)
        print('[Initialize Dataset]')

        # Initialize Parameters
        self.plti = config.data['plti']
        self.noise = config.data['noise']
        self.config = config

    def load_data(self):
        # Load Raw Data
        print('>> Load Dataset: ' + self.plti)
        raw_data = ascii.read(self.plti)
        # df = pd.read_csv('data/raw/plti/kplr_dr25_inj1_plti_sgmod.csv')
        df = raw_data.to_pandas()
        print(' > Data Shape: ' + str(df.shape))

        # Drop NA Data
        # df.dropna()

        # TODO: Check Recovered Output Value???? (for 2?)
        # df = df[df.Recovered != 2]
        # df = df.replace(2, 1)

        if self.config.data['drop_noise']: df = self._drop_noisy(df)

        # Extract Key Features
        print('>> Extract Features & Targets')
        X, y = self.extract_features(df)

        print(' > Features Extracted:')
        print(self.config.data['features'])
        print(' > X (shape): ' + str(X.shape))
        print(' > y (shape): ' + str(y.shape))

        print('-'*80)
        print('<Feature Engineering Process>')
        if self.config.data['feat_trans']: X = self._feat_trans(X)
        if self.config.data['ifs']: X = self._impute(X)
        if self.config.data['pca']: X = self._compute_pca(X)

        print('>> Final Feature Dimensions')
        print(' > X (shape): ' + str(X.shape))
        print(' > y (shape): ' + str(y.shape))

        return X, y

    def _feat_trans(self, X):
        # Feature Transformations
        print('>> Custom Feature Transformations')
        #                                                                 Baseline              [0.8573]
        # X['NTran'] = X['NTran'].transform(np.log)                     # Log Transform NTran   [0.8575]
        # X['depth'] = X['depth'].transform(lambda x: np.log(x+1))      # Log Transform Depth   [0.8576]
        # X['Rs'] = X['Rs'].transform(np.log)                           # Log Transform Rs      [0.8575]
        X['Ts'] = X['Ts'].transform(np.log)                             # Log Transform Ts      [0.8593]
        # X['a/Rs'] = X['a/Rs'].transform(np.log)                       # Log Transform a/Rs    [0.8574]
        # X['impact'] = X['impact'].transform(np.log)                    # Log Transform impact  [0.8574]
        X['SNR_DV'] = X['SNR_DV'].transform(lambda x: np.log(x+2))      # Log Transform impact  [0.8580]

        return X

    def _impute(self, X):
        print (">> Impute, Factorize, and Scale Features")
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = X[c].fillna(-1)
                X[c] = pd.factorize(X[c], sort=True)[0]
            if X[c].dtype == np.float64:
                X[c] = X[c].astype(np.float32)
                X[c] = X[c].fillna(-999)
                rscaler = StandardScaler()
                X[c] = rscaler.fit_transform(X[c].values.reshape(-1,1))

        return X

    def _compute_pca(self, X):
        print (">> Compute for the PCA Components and Add into Data Set")
        # PCA --> 85% = 13 | 90% = 26 | 95% = 57
        names = X.columns
        pca = PCA(n_components=X.shape[1])
        X_pca = pd.DataFrame(pca.fit_transform(X))
        X_pca.columns = ['pca' + str(i) for i in range(1, X.shape[1]+1)]
        names = names.append(X_pca.columns)
        X_all = pd.concat([X, X_pca], axis=1)

        return X_all

    def _drop_noisy(self, df):
        # FIXME: Issue with finding and dropping IDs...
        print (">> Drop Noisy KIC Sets")
        drop_id = open(self.config.data['noise'], 'r').read().split('\n')[10:-1]
        drop_id = list(map(lambda x: int(x.replace('\n', '').strip()), drop_id))

        # print(drop_id[0])
        for d in drop_id:
            if d in df.KIC: df.drop(d)

        print(' > Data Shape: ' + str(df.shape))
        return df

    def extract_features(self, df):
        # Augment Target Value Encoding
        df['Disp'].replace('FP', 0, inplace=True)
        df['Disp'].replace('PC', 1, inplace=True)

        X = df[self.config.data['features']]
        y = df['Disp']

        return X, y
