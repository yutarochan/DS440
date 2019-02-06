'''
TPS Classification Model Pipeline
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import time
import random
import argparse

from models import *
from utils.config import *
from utils.data import DataLoader

def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description='TPC Pipeline Efficiency Classification Pipeline')
    parser.add_argument('config', metavar='confg_json_file', default=None, help='File path to the configuration file.')
    args = parser.parse_args()

    # Parse Configuration File
    config, logger = parse_config(args.config)

    '''
    # Load Data
    # TODO: Just pass in the config and handle parameter parsing internally.
    print ("-" * 100)
    print ("Load Dataset")
    dl = DataLoader('data/raw/plti/kplr_dr25_inj1_plti.txt')
    X, y = dl.load_data()
    '''

    # Initialize Model Agent
    # agent = eval(config.agent).Agent(config)
    # agent.run()
    # agent.finalize()

if __name__ == '__main__':
    main()
