from autogluon.tabular import TabularPredictor
import pandas as pd

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'

# model paths
ba72 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/ba72"
d72 = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/d72"

train_val = pd.read_parquet('train_val.parquet')
test  = pd.read_parquet('test.parquet')

print("Number of rows, columns:", train_val.shape)
print("Number of rows, columns:", test.shape)

print("Number of unique PTIDs:", train_val['PTID'].nunique())
print("Number of unique PTIDs:", test['PTID'].nunique())

print(list(train_val.columns))

# 72 months
def month72(train_val):
    train_val = train_val.drop(columns=['p96', 'p120', 
                                        'd96', 'd120', 
                                        'bp72', 'bp96', 'bp120',
                                        'bd72', 'bd96', 'bd120'])
    diag_predictor = TabularPredictor(label='d72', problem_type='multiclass', eval_metric='f1_macro', path=d72).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )
    train_val = train_val.drop(columns='d72')
    ba_predictor = TabularPredictor(label='p72', problem_type='regression', eval_metric='root_mean_squared_error', path=ba72).fit(
        train_val, time_limit=300, presets='best_quality', num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto'
    )

    return ba_predictor, diag_predictor

if __name__ == "__main__":
    ba72_predictor, diag72_predictor = month72(train_val)
    ba72_predictor.fit_summary()
    diag72_predictor.fit_summary()