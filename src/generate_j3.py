#!/usr/bin/env python
from __future__ import division
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# from kaggler.data_io import load_data, save_data
# from kaggler.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from data_io import save_data

from const import ID_COL, TARGET_COL, SEED


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
    
    def create_new_ft(df):
        # Create new features
        df['lesion_size_ratio'] = df['tbp_lv_minorAxisMM'] / df['clin_size_long_diam_mm']
        df['lesion_shape_index'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2)
        df['hue_contrast'] = (df['tbp_lv_H'] - df['tbp_lv_Hext']).abs()
        df['luminance_contrast'] = (df['tbp_lv_L'] - df['tbp_lv_Lext']).abs()
        df['lesion_color_difference'] = np.sqrt(df['tbp_lv_deltaA'] ** 2 + df['tbp_lv_deltaB'] ** 2 + df['tbp_lv_deltaL'] ** 2)
        df['border_complexity'] = df['tbp_lv_norm_border'] + df['tbp_lv_symm_2axis']
        df['color_uniformity'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_radial_color_std_max'] + 0.0001)  # avoid division by zero

        df['position_distance_3d'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
        df['perimeter_to_area_ratio'] = df['tbp_lv_perimeterMM'] / df['tbp_lv_areaMM2']
        df['area_to_perimeter_ratio'] = df['tbp_lv_areaMM2'] / df['tbp_lv_perimeterMM']
        df['lesion_visibility_score'] = df['tbp_lv_deltaLBnorm'] + df['tbp_lv_norm_color']
        df['combined_anatomical_site'] = df['anatom_site_general'].astype(str) + '_' + df['tbp_lv_location'].astype(str)
        df['symmetry_border_consistency'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_norm_border']
        df['consistency_symmetry_border'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_norm_border'] / (df['tbp_lv_symm_2axis'] + df['tbp_lv_norm_border'] + 0.0001)

        df['log_lesion_area'] = np.log(df['tbp_lv_areaMM2'] + 1)
        df['normalized_lesion_size'] = df['clin_size_long_diam_mm'] / (df['age_approx'] + 0.0001)
        df['mean_hue_difference'] = (df['tbp_lv_H'] + df['tbp_lv_Hext']) / 2
        df['std_dev_contrast'] = np.sqrt((df['tbp_lv_deltaA'] ** 2 + df['tbp_lv_deltaB'] ** 2 + df['tbp_lv_deltaL'] ** 2) / 3)
        df['color_shape_composite_index'] = (df['tbp_lv_color_std_mean'] + df['tbp_lv_area_perim_ratio'] + df['tbp_lv_symm_2axis']) / 3
        df['lesion_orientation_3d'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])
        df['overall_color_difference'] = (df['tbp_lv_deltaA'] + df['tbp_lv_deltaB'] + df['tbp_lv_deltaL']) / 3

        # new features:
        df['color_consistency'] = df['tbp_lv_stdL'] / df['tbp_lv_Lext']
        df['consistency_color'] = df['tbp_lv_stdL'] * df['tbp_lv_Lext'] / (df['tbp_lv_stdL'] + df['tbp_lv_Lext'])
        df['size_age_interaction'] = df['clin_size_long_diam_mm'] * df['age_approx']
        df['hue_color_std_interaction'] = df['tbp_lv_H'] * df['tbp_lv_color_std_mean']
        df['lesion_severity_index'] = (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + df['tbp_lv_eccentricity']) / 3
        df['shape_complexity_index'] = df['border_complexity'] + df['lesion_shape_index']
        df['color_contrast_index'] = df['tbp_lv_deltaA'] + df['tbp_lv_deltaB'] + df['tbp_lv_deltaL'] + df['tbp_lv_deltaLBnorm']
        
        df['symmetry_perimeter_interaction'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_perimeterMM']
        df['comprehensive_lesion_index'] = (df['tbp_lv_area_perim_ratio'] + df['tbp_lv_eccentricity'] + df['tbp_lv_norm_color'] + df['tbp_lv_symm_2axis']) / 4
        df['color_variance_ratio'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_stdLExt']
        df['border_color_interaction'] = df['tbp_lv_norm_border'] * df['tbp_lv_norm_color']
        df['border_color_interaction_2'] = df['tbp_lv_norm_border'] * df['tbp_lv_norm_color'] / (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'])
        df['size_color_contrast_ratio'] = df['clin_size_long_diam_mm'] / df['tbp_lv_deltaLBnorm']
        df['age_normalized_nevi_confidence'] = df['tbp_lv_nevi_confidence'] / df['age_approx']
        df['age_normalized_nevi_confidence_2'] = ((df['clin_size_long_diam_mm']**2 + df['age_approx']**2) ** 0.5)
        df['color_asymmetry_index'] = df['tbp_lv_radial_color_std_max'] * df['tbp_lv_symm_2axis']
        
        df['volume_approximation_3d'] = df['tbp_lv_areaMM2'] * ((df['tbp_lv_x']**2 + df['tbp_lv_y']**2 + df['tbp_lv_z']**2) ** 0.5)
        df['color_range'] = (df['tbp_lv_L'] - df['tbp_lv_Lext']).abs() + (df['tbp_lv_A'] - df['tbp_lv_Aext']).abs() + (df['tbp_lv_B'] - df['tbp_lv_Bext']).abs()
        df['shape_color_consistency'] = df['tbp_lv_eccentricity'] * df['tbp_lv_color_std_mean']
        df['border_length_ratio'] = df['tbp_lv_perimeterMM'] / (2 * np.pi * (df['tbp_lv_areaMM2'] / np.pi) ** 0.5)
        df['age_size_symmetry_index'] = df['age_approx'] * df['clin_size_long_diam_mm'] * df['tbp_lv_symm_2axis']
        df['index_age_size_symmetry'] = df['age_approx'] * df['tbp_lv_areaMM2'] * df['tbp_lv_symm_2axis']        
        df['count_per_patient'] = df.groupby('patient_id')['isic_id'].transform('count')
    
        return df
    
    trn = create_new_ft(trn)
    tst = create_new_ft(tst)

    cat_cols = [x for x in trn.columns if trn[x].dtype == 'object' if x not in set(col_not_in_test + col_id + col_single + col_nul + [col_target])]
    num_cols = [x for x in trn.columns if trn[x].dtype != 'object' if x not in set(col_not_in_test + col_id + col_single + col_nul + [col_target])]
    cols = cat_cols + num_cols
    df = pd.concat([trn[cols], tst[cols]], axis=0)
    
    # num_cols = [i for i in num_cols if i not in ["tbp_lv_perimeterMM","tbp_lv_Lext","tbp_lv_norm_color", "tbp_lv_deltaLB"]]
    logging.info('categorical: {}, numerical: {}'.format(len(cat_cols), len(num_cols)))
    print(f"cat /n: {cat_cols}")
    print(f"num /n: {num_cols}")


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

