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

from const import N_FOLD, SEED
from data_io import load_data
from functools import partial
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def objective_func(train_file, trial):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 500, 1000),        
        "learning_rate": trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
        "min_child_weight": trial.suggest_int('min_child_weight', 16, 256),
        "colsample_bytree": trial.suggest_float('colsample_bytree', 0.4, 1.0),
        "subsample": trial.suggest_float('subsample', 0.4, 1.0),
        "lambda": trial.suggest_float('colsample_bytree', 0.4, 1.0),
        "random_state": SEED,
    }
    params.update({
        "objective":"reg:squarederror",
        "verbosity":0, # 0-3 silient to debug,
        "max_depth": trial.suggest_int('max_depth', 20, 80),
    })


    estimator = XGBRegressor(**params)
    X, y = load_data(train_file)    
    cv = KFold(5, shuffle=True)

    val_score = cross_val_score(
        estimator=estimator, 
        X=X, y=y, 
        cv=cv, 
        scoring="neg_root_mean_squared_error",
    )
    return np.mean(val_score)

def tuning(train_file, test_file, predict_valid_file, predict_test_file, n_est, n_stop, retrain):
    # Use functools.partial to include train_file in the objective function
    objective = partial(objective_func, train_file)

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"XGBoost tuning took {elapsed_time:.2f} seconds.")
    logging.info(f"best_params {study.best_params}")
    params = study.best_params

    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)
    train(params, X, y, X_tst, predict_valid_file, predict_test_file, n_est, n_stop, retrain)

 
def train(params, X, y, X_tst, predict_valid_file, predict_test_file, n_est, n_stop, retrain):
    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []

    cv = KFold(n_splits=N_FOLD)

    for i, (i_trn, i_val) in enumerate(cv.split(X), 1):
        logging.info('Training model #{}'.format(i))
        X_trn = X[i_trn,:]
        X_val = X[i_val,:]

        clf = XGBRegressor(**params)
        clf.fit(X_trn, 
                y[i_trn],
                eval_set=[(X_val, y[i_val])],
        )

        n_best = clf.best_iteration
        n_bests.append(n_best)
        
        p[i_val] = clf.predict(X_val)
        print(clf.best_score)
        best_score = clf.best_score
        # best_score=100
        logging.info(f'CV # {i}: best iteration={n_best}, best_score {best_score:.4f}')

        if not retrain:
            p_tst += clf.predict(X_tst) / N_FOLD

    logging.info(f'CV: {np.sqrt(mean_squared_error(y, p)):.4f}')
    logging.info(f'Saving validation predictions..')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')

def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est=100, n_leaf=200, lrate=.1, n_min=8, subcol=.8, subrow=.8,
                  n_stop=20, retrain=False):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    logging.info('Loading CV Ids')

    # params = {
    #     "n_estimators": n_est,        
    #     "learning_rate": lrate,
    #     "min_child_weight": n_min,
    #     "colsample_bytree": subcol,
    #     "subsample": subrow,
    #     "lambda": 1,
    #     "random_state": SEED,
    #     "early_stopping_rounds": n_stop,
    # }
    params = {'n_estimators': 593, 
              'learning_rate': 0.010683376267571213, 
              'min_child_weight': 221, 
              'colsample_bytree': 0.5111821154170035, 
              'subsample': 0.8545960483521913, 
              'max_depth': 66,
              "early_stopping_rounds": n_stop,
            }
    params.update({
        "objective":"reg:squarederror",
        "verbosity":0, # 0-3 silient to debug,
    })
    train(params, X, y, X_tst, predict_valid_file, predict_test_file, n_est, n_stop, retrain)

    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--n-leaf', type=int, dest='n_leaf')
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    # parser.add_argument('--subrow-freq', type=int, default=100, dest='subrow_freq')
    parser.add_argument('--n-min', type=int, default=1, dest='n_min')
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    model_name = str(os.path.splitext(os.path.splitext(os.path.basename(args.predict_test_file))[0])[0])
    print(f"model_name {model_name}")

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=f'logs/{model_name}.log')
    
    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  n_leaf=args.n_leaf,
                  lrate=args.lrate,
                  n_min=args.n_min,
                  subcol=args.subcol,
                  subrow=args.subrow,
                #   subrow_freq=args.subrow_freq,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    # logging.info("Tunning")
    # tuning(
    #     train_file=args.train_file,
    #     test_file = args.test_file,
    #     predict_valid_file=args.predict_valid_file,
    #     predict_test_file=args.predict_test_file,
    #     n_est=args.n_est,
    #     n_stop=args.n_stop,                  
    #     retrain=args.retrain
    # )   
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /60))
