#!/usr/bin/env python

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import argparse
import logging
import numpy as np
import os
import time

from const import N_FOLD, SEED
from data_io import load_data

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  retrain=False):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)
    print(f"before X {X.shape}, X_tst {X_tst.shape}")
    make_mae_rmse_diff = True
    if make_mae_rmse_diff:
        X_1 = X[:,X.shape[1]-1] - X[:,X.shape[1]-1-3]
        X_2 = X[:,X.shape[1]-2] - X[:,X.shape[1]-2-3]
        X_3 = X[:,X.shape[1]-3] - X[:,X.shape[1]-3-3]
        X_4 = X[:,X.shape[1]-1] - X[:,X.shape[1]-1-3-3]
        X_5 = X[:,X.shape[1]-2] - X[:,X.shape[1]-2-3-3]
        X_6 = X[:,X.shape[1]-3] - X[:,X.shape[1]-3-3-3]
        X_7 = X[:,X.shape[1]-1] - X[:,X.shape[1]-1-3-3-3]
        X_8 = X[:,X.shape[1]-2] - X[:,X.shape[1]-2-3-3-3]
        X_9 = X[:,X.shape[1]-3] - X[:,X.shape[1]-3-3-3-3]
        X = np.column_stack((X, X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9))

        X_1 = X_tst[:,X_tst.shape[1]-1] - X_tst[:,X_tst.shape[1]-1-3]
        X_2 = X_tst[:,X_tst.shape[1]-2] - X_tst[:,X_tst.shape[1]-2-3]
        X_3 = X_tst[:,X_tst.shape[1]-3] - X_tst[:,X_tst.shape[1]-3-3]
        X_4 = X_tst[:,X_tst.shape[1]-1] - X_tst[:,X_tst.shape[1]-1-3-3]
        X_5 = X_tst[:,X_tst.shape[1]-2] - X_tst[:,X_tst.shape[1]-2-3-3]
        X_6 = X_tst[:,X_tst.shape[1]-3] - X_tst[:,X_tst.shape[1]-3-3-3]
        X_7 = X_tst[:,X_tst.shape[1]-1] - X_tst[:,X_tst.shape[1]-1-3-3-3]
        X_8 = X_tst[:,X_tst.shape[1]-2] - X_tst[:,X_tst.shape[1]-2-3-3-3]
        X_9 = X_tst[:,X_tst.shape[1]-3] - X_tst[:,X_tst.shape[1]-3-3-3-3]
        X_tst = np.column_stack((X_tst,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9))


    print(f"after X {X.shape}, X_tst {X_tst.shape}")
    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])

    cv = KFold(N_FOLD, shuffle=True)

    # start training
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))

        model = Lasso(alpha=1.0)
        # Fit the model
        model.fit(X[i_trn], y[i_trn])

        p[i_val] = model.predict(X[i_val])
        best_score = np.sqrt(mean_squared_error(p[i_val], y[i_val]))
        logging.info(f'CV # {i}: best_score {best_score:.4f}')

        if not retrain:            
            p_tst += model.predict(X_tst) / N_FOLD

    logging.info(f'CV: {np.sqrt(mean_squared_error(y, p)):.4f}')
    logging.info(f'Saving validation predictions..')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')
    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    args = parser.parse_args()

    model_name = str(os.path.splitext(os.path.splitext(os.path.basename(args.predict_test_file))[0])[0])
    
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=f'logs/{model_name}.log')
    
    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file)
    # logging.info("Tunning..............")
    # tuning(
    #     train_file=args.train_file,
    #     test_file = args.test_file,
    #     predict_valid_file=args.predict_valid_file,
    #     predict_test_file=args.predict_test_file,
    #     early_stopping=args.early_stop,                  
    #     retrain=args.retrain
    # )   
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /60))
