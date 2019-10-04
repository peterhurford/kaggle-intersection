import sys

import pandas as pd
import numpy as np

from utils import print_step, rmse, run_cv_model, runLGB

from sklearn.model_selection import GroupKFold


print_step('Loading munged train')
train = pd.read_csv('processed_train.csv')
train_id = train['RowId']
train.drop('RowId', axis=1, inplace=True)

print_step('Loading munged test')
test = pd.read_csv('processed_test.csv')
test_id = test['RowId']
test.drop('RowId', axis=1, inplace=True)


print_step('Process targets')
target_data = {}
targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
           'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
           'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
           'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
           'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
           'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
           'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
for target in targets:
    if 'TimeFromFirstStop' not in target and '40' not in target and '60' not in target: # These targets don't count
        target_data[target] = train[target]
    train.drop(target, axis=1, inplace=True)
    test.drop(target, axis=1, inplace=True)


train_g = train.copy()
test_g = test.copy()
IS_OOFS_MODE = len(sys.argv) == 3 and sys.argv[2] == 'add_oofs'
if IS_OOFS_MODE:
    print_step('Loading OOFs 1/2')
    train_oofs = pd.read_csv('oofs_train.csv')
    test_oofs = pd.read_csv('oofs_test.csv')
    train = pd.concat((train, train_oofs), sort=False, axis=1).reset_index(drop=True)
    test = pd.concat((test, test_oofs), sort=False, axis=1).reset_index(drop=True)
    print_step('Loading OOFs 2/2')
    train_oofs = pd.read_csv('oofs_g_train.csv')
    test_oofs = pd.read_csv('oofs_g_test.csv')
    train_g = pd.concat((train_g, train_oofs), sort=False, axis=1).reset_index(drop=True)
    test_g = pd.concat((test_g, test_oofs), sort=False, axis=1).reset_index(drop=True)


model_id = sys.argv[1]
y = list(target_data.items())[int(model_id)]
label = y[0]
if IS_OOFS_MODE:
    label = label + '_w_oofs'
y = y[1]


cat_cols = [c for c in train.columns if '_NUNIQUE' not in c and '__COUNT' not in c and 'Match' not in c and '_from_' not in c and 'Has' not in c and c not in ['Latitude', 'Longitude', 'Hour', 'sin_Hour', 'cos_Hour', 'Weekend', 'Month', 'sin_Month', 'cos_Month', 'monthly_rainfall']]
split_cols = ['Path', 'IntersectionId', 'ExitStreetName', 'EntryStreetName',
              'Latitude', 'Longitude', 'EntryStreetType', 'ExitStreetType']
def is_ok_col(col):
    if '__COUNT' in col or '_NUNIQUE' in col:
        return True
    for split_col in split_cols:
        if split_col in col:
            return False
    return True
ok_cols = [c for c in train.columns.values if is_ok_col(c)]
print('Cat cols: {}'.format(cat_cols))
print('-')
print('Groupby Keeps: {}'.format(ok_cols))
print('-')
print('Groupby Drops: {}'.format([c for c in train.columns if c not in ok_cols]))
print('-')


print_step('Modeling {}'.format(label))
lgb_params = {'application': 'poisson',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 50,
              'learning_rate': 0.02,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.2,
              'verbosity': -1,
              'seed': 1,
              'lambda_l1': 0.1,
              'lambda_l2': 0.1,
              'max_delta_step': 0.7,
              'min_child_samples': 10,
              'min_child_weight': 5,
              'early_stop': 60,
              'verbose_eval': 30,
              'num_rounds': 1000,
              'num_threads': 8,
              'cat_cols': list(set(ok_cols) & set(cat_cols))}

if IS_OOFS_MODE:
    lgb_params['lambda_l1'] = 3.0
    lgb_params['lambda_l2'] = 3.0


split = GroupKFold(n_splits=5)
split = split.split(train_g, y, train_g['IntersectionId'])
results_g = run_cv_model(train_g[ok_cols], test_g[ok_cols], y, runLGB, lgb_params, rmse, label, n_folds=5, fold_splits=split)
lgb_params['cat_cols'] = cat_cols
results = run_cv_model(train, test, y, runLGB, lgb_params, rmse, label, n_folds=5)

imports_g = results_g['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(imports_g.sort_values('importance', ascending=False))

imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(imports.sort_values('importance', ascending=False))

import pdb
pdb.set_trace()

print_step('Saving OOFs 1/2')
pd.DataFrame({label: results['train']}).to_csv('{}_oof.csv'.format(label), index=False)
print_step('Saving OOFs 2/2')
pd.DataFrame({label: results_g['train']}).to_csv('{}_g_oof.csv'.format(label), index=False)
print_step('Saving preds 1/2')
pd.DataFrame({label: results['test']}).to_csv('{}_submit.csv'.format(label), index=False)
print_step('Saving preds 2/2')
pd.DataFrame({label: results_g['test']}).to_csv('{}_g_submit.csv'.format(label), index=False)
