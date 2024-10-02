#!/usr/bin/env python

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import cross_val_score
import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time
import optuna
from optuna.samplers import TPESampler

from const import N_FOLD, SEED, TARGET_COL
from data_io import load_data
# AutoML package:-
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_error

def train_predict(train_file, test_file, predict_valid_file, predict_test_file):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    make_mae_rmse_diff = True
    if make_mae_rmse_diff:
        X_1 = X[:,X.shape[1]-1] - X[:,X.shape[1]-1-3]
        X_2 = X[:,X.shape[1]-2] - X[:,X.shape[1]-2-3]
        X_3 = X[:,X.shape[1]-3] - X[:,X.shape[1]-3-3]
        X = np.column_stack((X, X_1,X_2,X_3))

        X_1 = X_tst[:,X_tst.shape[1]-1] - X_tst[:,X_tst.shape[1]-1-3]
        X_2 = X_tst[:,X_tst.shape[1]-2] - X_tst[:,X_tst.shape[1]-2-3]
        X_3 = X_tst[:,X_tst.shape[1]-3] - X_tst[:,X_tst.shape[1]-3-3]
        X_tst = np.column_stack((X_tst, X_1,X_2,X_3))


    model = TabularPredictor(
        label = TARGET_COL,
        problem_type = "regression",
        eval_metric  = "root_mean_squared_error",
        verbosity    = 0,
        path         = "../build/models/augl"
    )
    
    excluded_models = ["KNN",]
    presets        = "medium_quality"
    time_limit     = 60 * 60 * 2
    num_gpus       = 0

    train, train[TARGET_COL] = pd.DataFrame(X), y
    test = pd.DataFrame(X_tst)


    model.fit(train,
              excluded_model_types = excluded_models, 
              presets              = presets,
              time_limit           = time_limit,
              num_gpus             = num_gpus,
    )

    p = model.predict(train, as_pandas = False)    
    p_tst = model.predict(test, as_pandas = False)
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    

    args = parser.parse_args()

    model_name = str(os.path.splitext(os.path.splitext(os.path.basename(args.predict_test_file))[0])[0])
    logging.info(f"model_name {model_name}")

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=f'logs/{model_name}.log')
    
    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
    )
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /60))
