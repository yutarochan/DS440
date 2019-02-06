'''
Configuration Pipeline
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import os
import json
import uuid
from pprint import pprint
from easydict import EasyDict

from utils.scores import ScoreReport

def get_config(json_file):
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError as e:
            print("Invalid JSON File: " + e)
            exit(-1)

def save_config(fname, json_file):
    out = open(fname + '/config.json', 'w')
    out.write(json.dumps(json_file))
    out.close()

def parse_config(json_file):
    # Get Config Contents
    config, _ = get_config(json_file)

    # Generate Model UUID
    config.uuid = str(uuid.uuid4())

    try:
        # Initialize Model Artifact Folder
        config.output_dir = config.out_dir + '/' + config.model.replace(' ', '_').lower()
        if not os.path.exists(config.output_dir): os.makedirs(config.output_dir)

        # Initialize Metric Logging Object
        logger = ScoreReport(config.uuid, config.output_dir)
        config.output_dir = config.output_dir + '/' + config.uuid

        # Save Configuration File Under Log Directory
        save_config(config.output_dir, config)

        # Display Experiment Information
        print('='*80)
        print('Model: ' + config.model)
        print('Runtime UUID: ' + config.uuid)
        print('Output DIR: ' + config.output_dir)
        print('-'*80)
    except AttributeError:
        print("Missing parameter in config: model")
        exit(-1)

    # Display Configuration Setting
    print("[Configuration Parameters]\n")
    pprint(config)
    print('-'*80)

    return config, logger
