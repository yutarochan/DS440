'''
Feature Importance Aggregation
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os

from pyrankagg.rankagg import FullListRankAggregator

''' Load FI Ranks '''
def load_fi1(fname):
    data_raw = open(fname, 'r').read().split('\n')[1:-1]
    data = {}
    for i in data_raw:
        row = i.split(',')
        data[row[2]] = float(row[1])
    return data

def load_fi2(fname):
    data_raw = open(fname, 'r').read().split('\n')[1:-1]
    data = {}
    for i in data_raw:
        row = i.split(',')
        data[row[0]] = float(row[1])
    return data

# Load From Source
ada = load_fi1('data/eda/plti/_ada_feature_importance.csv')
ext = load_fi1('data/eda/plti/_ext_feature_importance.csv')
gbm = load_fi1('data/eda/plti/_gbm_feature_importance.csv')
las = load_fi2('data/eda/plti/_las_feature_selection.csv')
rfr = load_fi1('data/eda/plti/_rfr_feature_importance.csv')
rte = load_fi1('data/eda/plti/_rte_feature_importance.csv')
uni = load_fi2('data/eda/plti/_uni_feature_importance.csv')

# Aggregate Scores
score_agg = [ada, ext, gbm, las, rfr, rte, uni]

# Compute Rank Aggregation
FLRA = FullListRankAggregator()
aggRanks = FLRA.aggregate_ranks(score_agg)

print(aggRanks)
