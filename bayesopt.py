import sys
import warnings

import pandas as pd

from pprint import pprint
from bayes_opt import BayesianOptimization

from utils import print_step, rmse, run_cv_model, runLGB

from sklearn.model_selection import GroupKFold


print_step('Loading munged train')
train = pd.read_csv('processed_train.csv')
train_id = train['RowId']
train.drop('RowId', axis=1, inplace=True)


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


print('-')
print(train.shape)
print('-')
print_step('Dropping variables')
drops = pd.read_csv('drops.csv').drops.values
drops = [d for d in drops if d in train.columns]
train.drop(drops, axis=1, inplace=True)
print('-')
print('Dropped: {}'.format(drops))
print('-')
print(train.shape)
print('-')


train_g = train.copy()
IS_OOFS_MODE = len(sys.argv) == 3 and sys.argv[2] == 'add_oofs'
if IS_OOFS_MODE:
    print_step('Loading OOFs 1/2')
    train_oofs = pd.read_csv('oofs_train.csv')
    train = pd.concat((train, train_oofs), sort=False, axis=1).reset_index(drop=True)
    print_step('Loading OOFs 2/2')
    train_oofs = pd.read_csv('oofs_g_train.csv')
    train_g = pd.concat((train_g, train_oofs), sort=False, axis=1).reset_index(drop=True)


model_id = sys.argv[1]
y = list(target_data.items())[int(model_id)]
label = y[0]
if IS_OOFS_MODE:
    label = label + '_w_oofs'
y = y[1]


cat_cols = [c for c in train.columns if '_NUNIQUE' not in c and '__COUNT' not in c and 'Match' not in c and '_from_' not in c and 'Has' not in c and c not in ['Latitude', 'Longitude', 'Hour', 'sin_Hour', 'cos_Hour', 'Weekend', 'Month']]
split_cols = ['Path', 'IntersectionId', 'ExitStreetName', 'EntryStreetName',
              'Latitude', 'Longitude', 'EntryStreetType', 'ExitStreetType']
def is_ok_col(col):
    if '__COUNT' in col or '_NUNIQUE' in col:
        return True
    for split_col in split_cols:
        if split_col in col:
            return False
    return True
ok_cols = [c for c in train_g.columns.values if is_ok_col(c)]
print('-')
print('Cat cols: {}'.format(cat_cols))
print('-')
print('Groupby Keeps: {}'.format(ok_cols))
print('-')
print('Groupby Drops: {}'.format([c for c in train_g.columns if c not in ok_cols]))
print('-')
print(train_g[ok_cols].shape)
print('-')


def runBayesOpt(num_leaves, bag_fraction, feat_fraction, lambda1, lambda2, max_delta, min_child_samples, min_child_weight, min_cat, max_cat, cat_l2, cat_smooth, max_onehot, reg_or_poi):
    print('num_leaves {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, max_delta {}, min_child_samples {}, min_child_weight {}, min_cat {}, max_cat {}, cat_l2 {}, cat_smooth {}, max_onehot {}, reg_or_poi {}'.format(int(num_leaves), bag_fraction, feat_fraction, lambda1, lambda2, max_delta, int(min_child_samples), min_child_weight, int(min_cat), int(max_cat), cat_l2, cat_smooth, int(max_onehot), ['regression', 'poisson'][int(reg_or_poi)]))
    params = {'application': ['regression', 'poisson'][int(reg_or_poi)],
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': int(num_leaves),
              'learning_rate': 0.02,
              'bagging_fraction': bag_fraction,
              'feature_fraction': feat_fraction,
              'lambda_l1': lambda1,
              'lambda_l2': lambda2,
              'max_delta_step': max_delta,
              'min_child_samples': int(min_child_samples),
              'min_child_weight': min_child_weight,
              'early_stop': 60,
              'verbose_eval': 30,
              'verbosity': -1,
              'seed': 1,
              'nthread': 8,
              'num_rounds': 10000,
              'min_data_per_group': int(min_cat),
              'max_cat_threshold': int(max_cat),
              'cat_l2': cat_l2,
              'cat_smooth': cat_smooth,
              'max_cat_to_onehot': int(max_onehot),
              'cat_cols': list(set(ok_cols) & set(cat_cols))}
    split = GroupKFold(n_splits=5)
    split = split.split(train_g, y, train_g['IntersectionId'])
    results_g = run_cv_model(train_g[ok_cols], None, y, runLGB, params, rmse, label, n_folds=5, fold_splits=split)
    val_score = results_g['final_cv']
    print('score {}:num_leaves {}, bag_fraction {}, feat_fraction {}, lambda1 {}, lambda2 {}, max_delta {}, min_child_samples {}, min_child_weight {}, min_cat {}, max_cat {}, cat_l2 {}, cat_smooth {}, max_onehot {}, reg_or_poi {}'.format(val_score, int(num_leaves), bag_fraction, feat_fraction, lambda1, lambda2, max_delta, int(min_child_samples), min_child_weight, int(min_cat), int(max_cat), cat_l2, cat_smooth, int(max_onehot), ['regression', 'poisson'][int(reg_or_poi)]))
    return -val_score

LGB_BO = BayesianOptimization(runBayesOpt, {
    'num_leaves': (10, 200),
    'bag_fraction': (0.6, 1.0),
    'feat_fraction': (0.1, 0.9),
    'lambda1': (0, 10),
    'lambda2': (0, 10),
    'max_delta': (0, 10),
    'min_child_samples': (2, 100),
    'min_child_weight': (1, 100),
    'min_cat': (2, 200),
    'max_cat': (2, 100),
    'cat_l2': (1, 100),
    'cat_smooth': (1, 100),
    'max_onehot': (1, 100),
    'reg_or_poi': (0, 1)
})

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=8, n_iter=300, acq='ei', xi=0.0)

pprint(sorted([(r['target'], r['params']) for r in LGB_BO.res], reverse=True)[:3])
import pdb
pdb.set_trace()

# Fine tune
LGB_BO.set_bounds(new_bounds={'num_leaves': (20, 700),
                              'lambda1': (30, 300),
                              'lambda2': (30, 300),
                              'max_delta': (10, 300)})
LGB_BO.maximize(init_points=0, n_iter=5)
