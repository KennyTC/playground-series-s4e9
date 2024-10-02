#!/usr/bin/env python
import argparse
import json
import numpy as np
import os
import logging
from data_io import load_data
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')

    args = parser.parse_args()

    # y = np.loadtxt(args.target_file, delimiter=',')
    _, y = load_data(args.target_file)
    p = np.loadtxt(args.predict_file, delimiter=',')
   
    model_name = os.path.splitext(os.path.splitext(os.path.basename(args.predict_file))[0])[0]
    print(f'{model_name}\t{np.sqrt(mean_squared_error(y, p)):.6f}')
    
