from autogluon.tabular import TabularPredictor
import pandas as pd

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'

# model paths
ba120 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba120"
d120 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d120"

train_val = pd.read_parquet('train_val.parquet')
test  = pd.read_parquet('test.parquet')

print("Number of rows, columns:", train_val.shape)
print("Number of rows, columns:", test.shape)

print("Number of unique PTIDs:", train_val['PTID'].nunique())
print("Number of unique PTIDs:", test['PTID'].nunique())

print(list(train_val.columns))

# 120 months
def month120(train_val):
    train_val = train_val.drop(columns=['bp120',
                                        'bd120'])
    diag_predictor = TabularPredictor(label='d120', problem_type='multiclass', eval_metric='f1_macro', path=d120).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )
    train_val = train_val.drop(columns='d120')
    ba_predictor = TabularPredictor(label='p120', problem_type='regression', eval_metric='root_mean_squared_error', path=ba120).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )
    
    return ba_predictor, diag_predictor

if __name__ == "__main__":
    ba120_predictor, diag120_predictor = month120(train_val)
    ba120_predictor.fit_summary()
    diag120_predictor.fit_summary()