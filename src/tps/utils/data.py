'''
TPS Pipeline Efficiency Dataset
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import pandas as pd
from astropy.io import ascii

import warnings
warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        # Load Raw Data
        raw_data = ascii.read(self.filename)
        df = raw_data.to_pandas()
        df.dropna()

        X, y = self.preprocess(df)

        return X, y

    def drop_noisy(self, df):
        drop_id = open('data/raw/misc/DR25_DEModel_NoisyTargetList.txt', 'r').read().split('\n')[10:-1]
        drop_id = list(map(lambda x: str(x.replace('\n', '').strip()), drop_id))
        df.drop(drop_id[:10])
        return df

    def preprocess(self, df):
        # Drop Noisy Inputs
        # df = self.drop_noisy(df)
        df = df[df.Recovered != 2]

        # Select Features
        feat = ['Sky_Group', 'i_period', 'i_epoch', 'N_Transit', 'i_depth', 'i_dur', 'i_b', 'i_ror', 'i_dor']
        X = df[feat]
        y = df['Recovered']

        return X, y
