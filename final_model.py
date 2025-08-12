from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np


"""
this is the final model
predicts brain age in months using nested cv
"""

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'
# model save
model_folder = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/final_model"


def load_and_split(path):
    d = pd.read_csv(path, low_memory=False)
    a_cols = ['PTID', 'Date', 'Study', 'Phase', 'Sex', 'Age', 'Race', 'Diagnosis']
    b_cols = []
    for c in d.columns:
        if c.startswith("WMLS") or c.startswith("MUSE"):
            b_cols.append(c)
    d = d[a_cols + b_cols]
    d = d[d['Study'] == 'ADNI']
    counts = d['PTID'].value_counts()
    multi_ptids = counts[counts > 1].index
    d = d[d['PTID'].isin(multi_ptids)]

    d['Date'] = pd.to_datetime(d['Date'], infer_datetime_format=True)
    d = d.sort_values(['PTID','Date'])
    d_filtered = d

    # keep converters
    def keep_conversion(group):
        if group['Diagnosis'].iloc[0] != 'CN':
            return None
        if not group['Diagnosis'].isin(['AD', 'MCI']).any():
            return None
        return group
    
    # apply per PTID
    d_filtered = (d.groupby('PTID', group_keys=False).apply(keep_conversion).reset_index(drop=True))

    # convert calendar time to years
    d_filtered['calendar_time'] = (d_filtered.groupby('PTID')['Date'].transform(lambda x: ((x - x.iloc[0]).dt.days / 30).round(2)))

    d_filtered = d_filtered.drop(columns=['Date'])

    # print(d_filtered)
    # print(len(d_filtered))

    all_ptids = d_filtered['PTID'].unique()

    # splits patient ids into 60/20/20
    train_val_ptids, test_ptids = train_test_split(all_ptids, test_size=0.2, random_state=9)

    train_val = d_filtered[d_filtered['PTID'].isin(train_val_ptids)].reset_index(drop=True)
    test = d_filtered[d_filtered['PTID'].isin(test_ptids)].reset_index(drop=True)

    assert set(train_val_ptids).isdisjoint(test_ptids)

    train_val = train_val.drop(columns=['Study', 'Phase'])

    # print(train_val['Diagnosis'].value_counts(), test['Diagnosis'].value_counts())

    return train_val, test

def train_model(train_val, model_folder):
    predictor = TabularPredictor(label='calendar_time', problem_type='regression', eval_metric='root_mean_squared_error', 
                                 path=model_folder).fit(train_val, time_limit=480, presets='best_quality',
                                                        num_bag_folds=4, num_bag_sets=2, auto_stack=True, dynamic_stacking='auto')
    return predictor


if __name__ == "__main__":
    train_val, test = load_and_split(path)
    print(len(train_val))
    print(len(test))

    n_outer_folds = 5
    outer_gkf = GroupKFold(n_splits=n_outer_folds)
    outer_rmses = []

    for fold, (train_idx, val_idx) in enumerate(outer_gkf.split(train_val, groups=train_val['PTID']), start=1):
        df_train_outer = train_val.iloc[train_idx].reset_index(drop=True).drop(columns=['PTID'])
        df_val_outer = train_val.iloc[val_idx].reset_index(drop=True).drop(columns=['PTID'])

        predictor = TabularPredictor(label='calendar_time', problem_type='regression', eval_metric='root_mean_squared_error').fit(
            df_train_outer, time_limit=240, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto')
        
        res = predictor.evaluate(df_val_outer)
        outer_rmse = res['root_mean_squared_error']
        outer_rmses.append(outer_rmse)

    print("all rmse: ", (print(rmse) for rmse in outer_rmses))
    print("mean rmse: ", sum(outer_rmses)/len(outer_rmses))
    print("standard dev: ", np.std(outer_rmses))

    train_val = train_val.drop(columns=['PTID'])
    predictor = train_model(train_val, model_folder)
    predictor.fit_summary()

    print('\n')
    print('\n')
    print('\n')
    print('\n')

    print("all rmse: ", outer_rmses)
    print("mean rmse: ", sum(outer_rmses)/len(outer_rmses))
    print("standard dev: ", np.std(outer_rmses))
