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

def main():
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description='TPC Pipeline Efficiency Classification Pipeline')
    parser.add_argument('config', metavar='confg_json_file', default=None, help='File path to the configuration file.')
    args = parser.parse_args()

    # Parse Configuration File
    # config, wandb = parse_config(args.config)

if __name__ == '__main__':
    main()
