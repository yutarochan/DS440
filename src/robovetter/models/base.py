'''
Model Agent Base Interface
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function

class BaseAgent:
    def __init__(self, config):
        self.config = config

    def load_model(self, filename):
        raise NotImplementedError

    def save_model(self, filename):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def validate_model(self):
        raise NotImplementedError

    def hp_tune(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
