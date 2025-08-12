from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
import numpy as np
from final_model import load_and_split

# data path
path = '/Users/deanyao/Desktop/penn surf/mri_data/istaging_data.csv'
# model 
model_folder = "/Users/deanyao/Desktop/penn surf/mri_data/AutogluonModels/final_model"
# list of checkpoints
checkpoints = [24, 48, 72, 96, 120]

cp_dict = {24: ('p24', 'd24', 'bp24', 'bd24'),
           48: ('p48', 'd48', 'bp48', 'bd48'),
           72: ('p72', 'd72', 'bp72', 'bd72'),
           96: ('p96', 'd96', 'bp96', 'bd96'),
           120: ('p120', 'd120', 'bp120', 'bd120')}

def add_cols(path):
    train_val, test = load_and_split(path)
    # pred cols
    test['p24'] = np.nan
    test['p48'] = np.nan
    test['p72'] = np.nan
    test['p96'] = np.nan
    test['p120'] = np.nan
    # diag cols
    test['d24'] = np.nan
    test['d48'] = np.nan
    test['d72'] = np.nan
    test['d96'] = np.nan
    test['d120'] = np.nan
    # pred bool cols
    test['bp24'] = np.nan
    test['bp48'] = np.nan
    test['bp72'] = np.nan
    test['bp96'] = np.nan
    test['bp120'] = np.nan
    # bool diag cols
    test['bd24'] = np.nan
    test['bd48'] = np.nan
    test['bd72'] = np.nan
    test['bd96'] = np.nan
    test['bd120'] = np.nan

    # pred cols
    train_val['p24'] = np.nan
    train_val['p48'] = np.nan
    train_val['p72'] = np.nan
    train_val['p96'] = np.nan
    train_val['p120'] = np.nan
    # diag cols
    train_val['d24'] = np.nan
    train_val['d48'] = np.nan
    train_val['d72'] = np.nan
    train_val['d96'] = np.nan
    train_val['d120'] = np.nan
    # pred bool cols
    train_val['bp24'] = np.nan
    train_val['bp48'] = np.nan
    train_val['bp72'] = np.nan
    train_val['bp96'] = np.nan
    train_val['bp120'] = np.nan
    # bool diag cols
    train_val['bd24'] = np.nan
    train_val['bd48'] = np.nan
    train_val['bd72'] = np.nan
    train_val['bd96'] = np.nan
    train_val['bd120'] = np.nan
    return train_val, test


def interpolate_patient(times, values, t):
    """
    linear interpolation for time t
    if t is outside [times.min, times.max], do linear
    extrapolation using the two closest scans.
    """
    if t <= times.min():
        # extrapolate backwards using first two points
        x0, x1 = times.iloc[0], times.iloc[1]
        y0, y1 = values.iloc[0], values.iloc[1]
    elif t >= times.max():
        # extrapolate forward using last two points
        x0, x1 = times.iloc[-2], times.iloc[-1]
        y0, y1 = values.iloc[-2], values.iloc[-1]
    else:
        # true interpolation
        idx = np.searchsorted(times, t)
        x0, x1 = times.iloc[idx-1], times.iloc[idx]
        y0, y1 = values.iloc[idx-1], values.iloc[idx]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (t - x0)


def label_test(path):
    """
    labels test with the new vector field
    """
    _, test = add_cols(path)

    predictor = TabularPredictor.load(model_folder)

    test['ba_pred'] = predictor.predict(test)

    for pid, grp in test.groupby('PTID'):
        grp = grp.sort_values('calendar_time')
        # get true and pred cal time and diag vectors
        times = grp['calendar_time'].reset_index(drop=True)
        preds = grp['ba_pred'].reset_index(drop=True)
        diags = grp['Diagnosis'].reset_index(drop=True)
        # assert that they are the same length, they should be
        assert len(diags) == len(times) == len(preds)
        
        for cp in checkpoints:
            pcol, dcol, bpcol, bdcol = cp_dict[cp]

            # brain age value and flag
            if times.min() <= cp <= times.max():
                bp_flag = 0
                ba_cp = np.interp(cp, times, preds)
            else:
                bp_flag = 1
                ba_cp = interpolate_patient(times, preds, cp)
            
            idx_nearest = np.argmin(np.abs(times-cp))
            diag_cp = diags.iloc[idx_nearest]
            bd_flag = 0 if times.min() <= cp <= times.max() else 1

            test.loc[test["PTID"] == pid, pcol]  = ba_cp
            test.loc[test["PTID"] == pid, dcol]  = diag_cp
            test.loc[test["PTID"] == pid, bpcol] = bp_flag
            test.loc[test["PTID"] == pid, bdcol] = bd_flag
    
    test = test.drop(columns='ba_pred')
    # keep only baseline scans for each patient
    test = (test.sort_values(['PTID', 'calendar_time']).groupby('PTID', as_index=False).first())
    return test
        


def label_train_val(path):
    """
    trains on 0.64 and makes predictions on the 0.16
    labels train_val with the vector field
    """
    train_val, _ = add_cols(path)

    # create empty columns in train_val
    for cp in checkpoints:
        p,d,bp,bd = cp_dict[cp]
        train_val[[p,d,bp,bd]] = np.nan

    # out of fold brain age predictions for every scan
    oof_pred = pd.Series(index=train_val.index, dtype=float)
    gkf = GroupKFold(n_splits=5)

    for tr_idx, val_idx in gkf.split(train_val, groups=train_val['PTID']):
        model = TabularPredictor(label='calendar_time',
                                 problem_type='regression',
                                 eval_metric='root_mean_squared_error'
                                ).fit(train_val.iloc[tr_idx], time_limit=240, presets='best_quality',
                                                        num_bag_folds=4, num_bag_sets=1, auto_stack=True, dynamic_stacking='auto')
        oof_pred.iloc[val_idx] = model.predict(train_val.iloc[val_idx])

    train_val["ba_pred"] = oof_pred

    # interpolate and extrapolate per patient
    for pid, grp in train_val.groupby('PTID'):
        grp = grp.sort_values('calendar_time')
        times = grp['calendar_time'].reset_index(drop=True)
        preds = grp['ba_pred'].reset_index(drop=True)
        diags = grp['Diagnosis'].reset_index(drop=True)

        for cp in checkpoints:
            pcol, dcol, bpcol, bdcol = cp_dict[cp]

            if times.min() <= cp <= times.max():
                bp_flag = 0
                ba_cp  = np.interp(cp, times, preds)
            else:
                bp_flag = 1
                ba_cp  = interpolate_patient(times, preds, cp)

            idx_nearest = np.argmin(np.abs(times-cp))
            diag_cp = diags.iloc[idx_nearest]
            bd_flag = 0 if times.min() <= cp <= times.max() else 1

            train_val.loc[train_val["PTID"] == pid, pcol]  = ba_cp
            train_val.loc[train_val["PTID"] == pid, dcol]  = diag_cp
            train_val.loc[train_val["PTID"] == pid, bpcol] = bp_flag
            train_val.loc[train_val["PTID"] == pid, bdcol] = bd_flag
    
    train_val = train_val.drop(columns='ba_pred')

    # keep only baseline row per patient
    train_val = (train_val.sort_values(['PTID', 'calendar_time']).groupby('PTID', as_index=False).first())

    return train_val


def label_data(path):
    train_val = label_train_val(path)
    test = label_test(path)
    return train_val, test

if __name__ == "__main__":
    train_val, test = load_and_split(path)
    # number of rows, columns
    print("Number of rows, columns:", train_val.shape)
    print("Number of rows, columns:", test.shape)

    # number of unique patient IDs
    print("Number of unique PTIDs:", train_val['PTID'].nunique())
    print("Number of unique PTIDs:", test['PTID'].nunique())

    tv, t = label_data(path)
    # number of rows, columns
    print("Number of rows, columns:", tv.shape)
    print("Number of rows, columns:", t.shape)

    # number of unique patient IDs
    print("Number of unique PTIDs:", tv['PTID'].nunique())
    print("Number of unique PTIDs:", t['PTID'].nunique())
    tv.to_parquet('train_val.parquet', index=False)
    t.to_parquet('test.parquet',  index=False)