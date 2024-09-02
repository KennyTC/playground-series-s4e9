#!/usr/bin/env python
import argparse
import json
import numpy as np
import os
from metric import pauc_80
import logging
from data_io import load_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')

    args = parser.parse_args()

    # y = np.loadtxt(args.target_file, delimiter=',')
    _, y = load_data(args.target_file)
    p = np.loadtxt(args.predict_file, delimiter=',')
   
    model_name = os.path.splitext(os.path.splitext(os.path.basename(args.predict_file))[0])[0]
    print(f'{model_name}\t{pauc_80(y, p):.6f}')
    
