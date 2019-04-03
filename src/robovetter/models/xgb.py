'''
Xtreeme Gradient Boosting Classifier Model Agent
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import numpy as np
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

from models.base import BaseAgent
from utils.data import DataLoader
from utils.scores import ScoreReport

class Agent(BaseAgent):
    def __init__(self, config, logger):
        self.config = config
        self.report = logger

    def run(self):
        # Load Dataset
        dl = DataLoader(self.config)
        X, y = dl.load_data()

        # Train Model - No Training Required, Just KF Validation
        self.train(X, y)

    def train(self, X, y):
        # Initialize KF-Cross Validation
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X.values):
            print('> K-FOLD Iteration')

            # Extract Train/Test Data
            X_train, X_valid = X.values[train_index], X.values[test_index]
            Y_train, Y_valid = y[train_index], y[test_index]

            # Train Model
            model = self.train_model(X_train, Y_train)

            # Validate Model
            y_actual, y_binary, y_prob = self.validate_model(model, X_valid, Y_valid)
            self.report.append_result(y_actual, y_binary, y_prob)

    def train_model(self, X, y):
        # XGB Parameters
        xgb_params = {
            'eta': 0.01,                   # 0.01
            'max_depth': 6,                # 6
            'min_child_weight': 1,
            'subsample': 0.80,              # 0.80
            'objective': 'binary:logitraw',
            'colsample_bytree': 0.50,
            'scale_pos_weight': 2,
            'eval_metric': 'auc',
            'base_score': np.mean(y),
            'gpu_id': 0,
            'seed': 9389493,
            'silent': 1
        }

        model = XGBClassifier()
        # xgb_train = xgb.DMatrix(X, y)
        # model = xgb.train(xgb_params, xgb_train)
        model.fit(X, y)
        return model

    def validate_model(self, model, X, y):
        y_prob = model.predict_proba(X)
        y_binary = [0 if p[1] <= 0.5 else 1 for p in y_prob]

        return y, y_binary, y_prob[:, 1]

    def finalize(self):
        # Finalize Report
        self.report.generate_rocplot()
        self.report.generate_score_report()
