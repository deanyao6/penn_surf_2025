from autogluon.tabular import TabularPredictor
import pandas as pd

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'

# model paths
ba96 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba96"
d96 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d96"

train_val = pd.read_parquet('train_val.parquet')
test  = pd.read_parquet('test.parquet')

print("Number of rows, columns:", train_val.shape)
print("Number of rows, columns:", test.shape)

print("Number of unique PTIDs:", train_val['PTID'].nunique())
print("Number of unique PTIDs:", test['PTID'].nunique())

print(list(train_val.columns))

# 96 months
def month96(train_val):
    train_val = train_val.drop(columns=['p120', 
                                        'd120', 
                                        'bp96', 'bp120',
                                        'bd96', 'bd120'])
    diag_predictor = TabularPredictor(label='d96', problem_type='multiclass', eval_metric='f1_macro', path=d96).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )
    train_val = train_val.drop(columns='d96')
    ba_predictor = TabularPredictor(label='p96', problem_type='regression', eval_metric='root_mean_squared_error', path=ba96).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )

    return ba_predictor, diag_predictor

if __name__ == "__main__":
    ba96_predictor, diag96_predictor = month96(train_val)
    ba96_predictor.fit_summary()
    diag96_predictor.fit_summary()