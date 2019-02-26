'''
Voting Ensemble Model (Top 3)
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier

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
            # Extract Train/Test Data
            X_train, X_valid = X.values[train_index], X.values[test_index]
            Y_train, Y_valid = y[train_index], y[test_index]

            # Train Model
            model = self.train_model(X_train, Y_train)

            # Validate Model
            y_actual, y_binary, y_prob = self.validate_model(model, X_valid, Y_valid)
            self.report.append_result(y_actual, y_binary, y_prob)

    def train_model(self, X, y):
        print('>> KFOLD Iteration <<')
        # Define Models
        m1 = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
        m2 = AdaBoostClassifier(n_estimators=500)
        m3 = XGBClassifier()

        model = VotingClassifier(estimators=[('cat', m1), ('ada', m2), ('xgb', m3)], voting='soft')
        model = model.fit(X, y)
        return model

    def validate_model(self, model, X, y):
        y_prob = model.predict_proba(X)
        y_binary = [0 if p[1] <= 0.5 else 1 for p in y_prob]

        return y, y_binary, y_prob[:, 1]

    def finalize(self):
        # Finalize Report
        self.report.generate_rocplot()
        self.report.generate_score_report()
