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
    logging.info(f'loading raw data: train {trn.shape}, test {tst.shape}')
    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]
    trn.drop(TARGET_COL, axis=1, inplace=True)

    cat_cols = [x for x in trn.columns if trn[x].dtype == 'object']
    num_cols = [x for x in trn.columns if trn[x].dtype != 'object']


    col_not_in_test = ['iddx_4',
    'iddx_1',
    'iddx_full',
    'iddx_3',
    'mel_mitotic_index',
    'iddx_5',
    'mel_thick_mm',
    'lesion_id',
    'tbp_lv_dnn_lesion_confidence',
    'target',
    'iddx_2'
    ]
    col_target = 'target'
    col_id = [
        "isic_id", # unique id
    ]
    col_single = [
        "image_type", # single value
    ]
    col_nul = [
        "iddx_2","iddx_3","iddx_4","iddx_5", # almost nan
        "iddx_full", # contains same information as column iddx_1, so I decide not to use
        "lesion_id", # 95% nan,        
        "mel_mitotic_index", # almost nan, 
        "mel_thick_mm", # 99% nan   
    ]
    cat_cols = [x for x in trn.columns if trn[x].dtype == 'object' if x not in set(col_not_in_test + col_id + col_single + col_nul + [col_target])]
    num_cols = [x for x in trn.columns if trn[x].dtype != 'object' if x not in set(col_not_in_test + col_id + col_single + col_nul + [col_target])]
    num_cols = [i for i in num_cols if i not in ["tbp_lv_perimeterMM","tbp_lv_Lext","tbp_lv_norm_color", "tbp_lv_deltaLB"]]
    logging.info('categorical: {}, numerical: {}'.format(len(cat_cols),len(num_cols)))
    print(f"cat: {cat_cols}")
    print(f"num: {num_cols}")

    cols = cat_cols + num_cols
    df = pd.concat([trn[cols], tst[cols]], axis=0)
    logging.info(f"df shape {df.shape}")

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

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

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

