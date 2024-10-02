#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

# from kaggler.data_io import load_data, save_data
# from kaggler.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from data_io import save_data
from const import ID_COL, TARGET_COL


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    trn["milage"]= trn["milage"].str.replace("mi.","").replace(" ","").str.replace(",","").astype(int, errors="ignore")
    trn["price"]= trn["price"].str.replace("$","").replace(" ","").str.replace(",","").astype(int, errors="ignore")
    
    logging.info(f'loading raw data: train {trn.shape}, test {tst.shape}')
    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]
    trn.drop(TARGET_COL, axis=1, inplace=True)

    
    cat_cols = [x for x in trn.columns if trn[x].dtype == 'object']
    num_cols = [x for x in trn.columns if trn[x].dtype != 'object']

    cols = cat_cols + num_cols
    df = pd.concat([trn[cols], tst[cols]], axis=0)
    logging.info(f"df shape {df.shape}, cat {cat_cols}, num {num_cols}")

    logging.info('label encoding categorical variables')
    lbe = LabelEncoder()
    for col in cat_cols:
        df[col] = lbe.fit_transform(df[col])
    df[num_cols] = df[num_cols].fillna(-1)

    # print(f"feature_map_file {feature_map_file}")
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(df.columns):
            f.write('{}\t{}\t\n'.format(i, col))

    logging.info('saving features')
    save_data(df.values[:n_trn,], y, train_feature_file)
    save_data(df.values[n_trn:,], None, test_feature_file)
    
if __name__ == '__main__':
    logname = f"{os.path.basename(__file__).replace('generate_','').replace('.py','')}.log"
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename=f'logs/{logname}'
                    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

