import sys

import pandas as pd
import numpy as np

from utils import print_step, rmse, run_cv_model, runLGB

from sklearn.preprocessing import LabelEncoder
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


print_step('Label encode')
cat_cols = train.dtypes[(train.dtypes != np.float) & (train.dtypes != np.int64)]
cat_cols = list(cat_cols.keys())

for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]]))
    train.loc[:, c] = le.transform(train[c])
    test.loc[:, c] = le.transform(test[c])


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

split_cols = ['EntryStreetExitStreet', 'Path', 'PathIntersection', 'IntersectionId',
              'ExitStreetName', 'EntryStreetName', 'EntryStreetNameHeading',
              'ExitStreetNameHeading', 'Latitude', 'Longitude', 'EntryStreetNameTurn',
              'ExitStreetNameTurn', 'PathNameTurn', 'PathType', 'PathTypeTurn']


print_step('Modeling {}'.format(label))
lgb_params = {'application': 'poisson',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 50,
              'learning_rate': 0.02,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.3,
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
              'num_threads': 6,
              'cat_cols': list(set(cat_cols) - set(split_cols))}

if IS_OOFS_MODE:
    lgb_params['lambda_l1'] = 3.0
    lgb_params['lambda_l2'] = 3.0


split = GroupKFold(n_splits=5)
split = split.split(train_g, y, train_g['IntersectionId'])
results_g = run_cv_model(train_g.drop(split_cols, axis=1), test_g.drop(split_cols, axis=1), y, runLGB, lgb_params, rmse, label, n_folds=5, fold_splits=split)
lgb_params['cat_cols'] = cat_cols
results = run_cv_model(train, test, y, runLGB, lgb_params, rmse, label, n_folds=5)
import pdb
pdb.set_trace()

imports = results_g['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(imports.sort_values('importance', ascending=False))

imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(imports.sort_values('importance', ascending=False))

print_step('Saving OOFs 1/2')
pd.DataFrame({label: results['train']}).to_csv('{}_oof.csv'.format(label), index=False)
print_step('Saving OOFs 2/2')
pd.DataFrame({label: results_g['train']}).to_csv('{}_g_oof.csv'.format(label), index=False)
print_step('Saving preds 1/2')
pd.DataFrame({label: results['test']}).to_csv('{}_submit.csv'.format(label), index=False)
print_step('Saving preds 2/2')
pd.DataFrame({label: results_g['test']}).to_csv('{}_g_submit.csv'.format(label), index=False)
