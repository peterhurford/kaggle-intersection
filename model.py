import sys

import pandas as pd
import numpy as np

from utils import print_step, rmse, run_cv_model, runLGB

from sklearn.preprocessing import LabelEncoder


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


print_step('Label encode')
cat_cols = train.dtypes[(train.dtypes != np.float) & (train.dtypes != np.int64)]
cat_cols = list(cat_cols.keys())

for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]]))
    train.loc[:, c] = le.transform(train[c])
    test.loc[:, c] = le.transform(test[c])



model_id = sys.argv[1]
y = list(target_data.items())[int(model_id)]
label = y[0]
y = y[1]
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
              'num_threads': 5,
              'cat_cols': cat_cols}

results = run_cv_model(train, test, y, runLGB, lgb_params, rmse, label, n_folds=5)
import pdb
pdb.set_trace()

print_step('Saving OOFs')
pd.DataFrame({label: results['train']}).to_csv('{}_oof.csv'.format(label), index=False)
print_step('Saving preds')
pd.DataFrame({label: results['test']}).to_csv('{}_submit.csv'.format(label), index=False)
