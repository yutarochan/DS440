'''
Baseline Model Agent
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from scipy.integrate import quad

from models.base import BaseAgent
from utils.data import DataLoader
from utils.scores import ScoreReport

def integrand(t, a, b):
    return np.exp(-t/b) * (t ** (a-1))

class GammaModel:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def pred(self, X):
        predList = []
        for x in X:
            pred = self.c * quad(integrand, 0, x, args=(self.a, self.b))[0] / ((self.b ** self.a) * np.random.gamma(self.a))
            predList.append(pred)

        return predList

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

        # Generate Predictions
        self.predict(X, y)

    def train(self, X, y):
        # Initialize KF-Cross Validation
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X.values):
            # Extract Training Data
            X_train = np.take(X.values, train_index)
            Y_train = np.take(y.values, train_index)

            # Train Model
            model = self.train_model(X_train, Y_train)

            # Validation Inference
            X_valid = np.take(X.values, test_index)
            Y_valid = np.take(y.values, test_index)

            y_actual, y_binary, y_prob = self.validate_model(model, X_valid, Y_valid)
            self.report.append_result(y_actual, y_binary, y_prob)

    def train_model(self, X, y):
        return GammaModel(self.config.a, self.config.b, self.config.c)

    def validate_model(self, model, X, y):
        y_prob = model.pred(X)
        y_binary = [0 if p <= 0.5 else 1 for p in y_prob]
        return y, y_binary, y_prob

    def predict(self, X, y):
        model = GammaModel(self.config.a, self.config.b, self.config.c)
        y_prob = model.pred(X)
        y_binary = [0 if p <= 0.5 else 1 for p in y_prob]

        return y, y_binary, y_prob

    def finalize(self):
        # Finalize Report
        self.report.generate_rocplot()
        self.report.generate_score_report()
