#!/usr/bin/env python

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

import argparse
import logging
import numpy as np
import operator
import os
import pandas as pd
import time

from const import N_FOLD, SEED
from data_io import load_data
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor, Pool
from functools import partial
import optuna
from optuna.samplers import TPESampler

# def objective_cb(train_file, trial):
#     params = {
#         'loss_function':     'RMSE',
#         'iterations':        200,
#         'verbose':           False,
#         'random_state':      SEED,
#         'learning_rate':     trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
#         'max_depth':         trial.suggest_int('max_depth', 4, 8),
#         'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
#         'subsample':         trial.suggest_float('subsample', 0.4, 1.0),
#         'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
#         'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 5, 100),
#         # 'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 0.8, 4.0),
#         # 'bootstrap_type':    'Bayesian',  # Optional: depending on your use case, you may want to tune this as well
#     }

#     estimator = CatBoostRegressor(**params)
#     X, y = load_data(train_file)    
    
#     cv = KFold(N_FOLD, shuffle=True)

#     val_score = cross_val_score(
#         estimator=estimator, 
#         X=X, y=y, 
#         cv=cv, 
#         scoring="neg_root_mean_squared_error",
#     )
#     return np.mean(val_score)

# def tuning(train_file, test_file, predict_valid_file, predict_test_file,early_stopping, retrain):
#     # Use functools.partial to include train_file in the objective function
#     objective = partial(objective_cb, train_file)

#     start_time = time.time()
#     study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
#     study.optimize(objective, n_trials=100, show_progress_bar=True)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     logging.info(f"Catboost tuning took {elapsed_time:.2f} seconds.")

#     # Output the best parameters and the best score achieved
#     logging.info(f"Best parameters: {study.best_params}")
#     logging.info(f"Best score achieved: {study.best_value}")
#     logging.info("Train CV again using best params-------------------")

#     params = study.best_params.update({
#         'loss_function':     'RMSE',
#         'verbose':           1000,
#         'random_state':      SEED
#     })

#     X, y = load_data(train_file)
#     X_tst, _ = load_data(test_file)
#     print(f"params {params}")
#     train(params, X, y, X_tst, predict_valid_file, predict_test_file, early_stopping, retrain)

def train(params, X, y, cat_features, X_tst, predict_valid_file, predict_test_file, n_stop, retrain):
    '''
    start training
    '''
    p = np.zeros(X.shape[0])
    p_tst = np.zeros(X_tst.shape[0])
    n_bests = []

    cv = KFold(N_FOLD, shuffle=True)

    # start training
    for i, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
        logging.info('Training model #{}'.format(i))

        trn_data = Pool(X[i_trn], label=y[i_trn], cat_features=cat_features)
        val_data = Pool(X[i_val], label=y[i_val], cat_features=cat_features)

        model = CatBoostRegressor(**params)

        model.fit(trn_data,
                  eval_set=val_data,
                  use_best_model=True,
                  early_stopping_rounds=n_stop,
                )
        n_best = model.get_best_iteration()
        n_bests.append(n_best)
        
        p[i_val] = model.predict(X[i_val])
        best_score = np.sqrt(mean_squared_error(p[i_val], y[i_val]))
        logging.info(f'CV # {i}: best iteration={n_best}, best_score {best_score:.4f}')

        if not retrain:            
            p_tst += model.predict(X_tst) / N_FOLD

    logging.info(f'CV: {np.sqrt(mean_squared_error(y, p)):.4f}')
    logging.info(f'Saving validation predictions..')
    np.savetxt(predict_valid_file, p, fmt='%.6f', delimiter=',')
    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f', delimiter=',')



def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                  n=100, learning_rate=.1, max_depth=4, l2_leaf_reg=0.4, sub_sample=0.4, 
                  colsample_bylevel = 0.4, min_data_in_leaf =4, scale_pos_weight = 0.8,
                  n_stop = 20, retrain=False):

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    feature_map_file = pd.read_csv(feature_map_file, sep="\t", header=None)
    cat_features = feature_map_file.loc[feature_map_file[2]=="obj",0].to_list()
    X_new = np.empty(X.shape, dtype=object)
    X_new_tst = np.empty(X_tst.shape, dtype=object)
    for i in range(X.shape[1]):
        if i in cat_features:  
            X_new[:, i] = X[:, i].astype(int)
            X_new_tst[:, i] = X_tst[:, i].astype(int)
        else:
            X_new[:, i] = X[:, i]
            X_new_tst[:,i] = X_tst[:,i]
    X = X_new
    X_tst = X_new_tst

  
    params = {
        'loss_function':     'MAE',
        'iterations':        n,
        'verbose':           100,
        'random_state':      SEED,
        'learning_rate':     learning_rate,
        'max_depth':         max_depth,
        'l2_leaf_reg':       l2_leaf_reg,
        'min_data_in_leaf':  min_data_in_leaf,
        # 'scale_pos_weight':  scale_pos_weight,
        # 'bootstrap_type':    'Bayesian',  
    }
    
    train(params, X, y, cat_features, X_tst, predict_valid_file, predict_test_file, n_stop, retrain)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--n', type=int, dest='n')
    parser.add_argument('--learning_rate', type=float, dest="learning_rate")
    parser.add_argument('--max_depth', type=float, default=1)
    parser.add_argument('--l2_leaf_reg', type=int)
    parser.add_argument('--min_data_in_leaf', type=float)
    parser.add_argument('--scale_pos_weight', type=int)
    parser.add_argument('--early-stop', type=int)
    parser.add_argument('--retrain', default=False, action='store_true')

    args = parser.parse_args()

    model_name = str(os.path.splitext(os.path.splitext(os.path.basename(args.predict_test_file))[0])[0])
    
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=f'logs/{model_name}.log')
    
    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file = args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n=args.n,
                  learning_rate=args.learning_rate,
                  max_depth=args.max_depth,
                  l2_leaf_reg=args.l2_leaf_reg,
                  scale_pos_weight=args.scale_pos_weight,
                  n_stop=args.early_stop,                  
                  retrain=args.retrain)
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
