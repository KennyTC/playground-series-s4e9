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

    # trn["milage"]= trn["milage"].str.replace("mi.","").replace(" ","").str.replace(",","").astype(int, errors="ignore")
    # trn["price"]= trn["price"].str.replace("$","").replace(" ","").str.replace(",","").astype(int, errors="ignore")
    
    logging.info(f'loading raw data: train {trn.shape}, test {tst.shape}')
    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]
    

    def ft_eng(df):       
        df['brand_model'] = df['brand'] + '_' + df['model']        
        # Extract horsepower from engine descriptions
        df['horsepower'] = df['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
        df['horsepower_category'] = df['horsepower'].apply(lambda x: 1 if x <=1 else (3 if x>=3 else 2)).astype("category")
        df['power_to_fuel_type'] = df['horsepower'].astype(str) + '_' + df['fuel_type']
        # Calculate age of vehicle
        current_year = pd.Timestamp('now').year
        df['vehicle_age'] = current_year - df['model_year']
        df['vehicle_age_x_horsepower'] = df["horsepower"] * df['model_year']
        df['vehicle_age_horsepower_mean'] = df[["horsepower",'model_year']].mean(axis=1)
        df['mileage_per_year'] = df['milage'] / (current_year - df['model_year'] + 1)
        df['color_combo'] = df['ext_col'] + '_' + df['int_col']
        # Create fuel economy category based on fuel type
        fuel_categories = {
            'Gasoline': 'Fossil',
            'Diesel': 'Fossil',
            'Hybrid': 'Hybrid',
            'Electric': 'Electric',
            'E85 Flex Fuel': 'Flex'
        }
        df['fuel_category'] = df['fuel_type'].map(fuel_categories).astype("category")
        # Convert accident history to binary
        df['has_accident'] = (df['accident'] != 'None reported').astype(int)
        # Clean title binary conversion
        df['is_title_clean'] = (df['clean_title'] == 'Yes').astype(int)
        df['accident_title_status'] = df['accident'].astype(str) + '_' + df['clean_title'].astype(str)
        # df['engine_transmission'] = df['engine'].str.extract(r'(\d+\.?\d*)L') + '_' + df['transmission']
        df['model_year_fuel_type'] = df['model_year'].astype(str) + '_' + df['fuel_type']
        # Simplify transmission types
        transmission_categories = {
            'Automatic': 'Automatic',
            'Manual': 'Manual',
            'A/T': 'Automatic',
            'M/T': 'Manual'
        }
        df['transmission_type'] = df['transmission'].replace(transmission_categories).fillna('Other').astype("category")
        df['milage_impact1'] = df['milage'].apply(lambda x: 1 if x < 20000 else 0).astype("category")
        df['milage_impact2'] = df['milage'].apply(lambda x: 20000 - x if x < 20000 else 0).astype("category")

        df['model_year_impact1'] = df['model_year'].apply(lambda x: 1 if (x >= 2001 & x <= 2023) else 0).astype("category")
        df['model_year_impact2'] = df['model_year'].apply(lambda x: 2023 - x if (x >= 2001 & x <= 2023) else 0).astype("category")        
        df["model_year"] = df["model_year"].astype("category")
        # num_cols = [x for x in trn.columns if trn[x].dtype != 'object' and x != TARGET_COL]
        # print(num_cols)
        # df["target_transf"] = df[TARGET_COL] - df[num_cols].mean(axis=1) * 0.1
        # df["sum"] = df[num_cols].sum(axis=1)
        # df["sum_std"] = df.groupby("sum")[TARGET_COL].std()

        return df

    trn = ft_eng(trn)
    tst = ft_eng(tst)
    
    trn.drop(TARGET_COL, axis=1, inplace=True)

    cat_cols = [x for x in trn.columns if trn[x].dtype == 'object']
    num_cols = [x for x in trn.columns if trn[x].dtype != 'object']

    cols = cat_cols + num_cols
    df = pd.concat([trn[cols], tst[cols]], axis=0)
    logging.info(f"df shape {df.shape}, cat {cat_cols}, num {num_cols}")

    # logging.info('label encoding categorical variables')
    # lbe = LabelEncoder()
    # for col in cat_cols:
    #     df[col] = lbe.fit_transform(df[col])
    # df[num_cols] = df[num_cols].fillna(-1)
    RARE = []
    for c in cat_cols:
        # LABEL ENCODE
        df[c],_ = df[c].factorize()
        df[c] -= df[c].min()
        vc = df[c].value_counts()
        
        # IDENTIFY RARE VALUES
        RARE.append( vc.loc[vc<40].index.values )
        n = df[c].nunique()
        mn = df[c].min()
        mx = df[c].max()
        r = len(RARE[-1])
        print(f'{c}: nunique={n}, min={mn}, max={mx}, rare_ct={r}')
        
        # RELABEL RARE VALUES AS ZERO
        df[c] += 1
        df.loc[df[c].isin(RARE[-1]),c] = 0 


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

