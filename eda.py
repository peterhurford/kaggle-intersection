import sys
import random

import pandas as pd
import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split

from utils import print_step


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


X_train, X_test = train_test_split(train, test_size=0.2)
cat_cols = train.dtypes[(train.dtypes != np.float) & (train.dtypes != np.int64)]
cat_cols = list(cat_cols.keys())
test_n = test.shape[0]
x_test_n = X_test.shape[0]
for cat_col in cat_cols:
    in_test_only = set(test[cat_col]) - set(train[cat_col])
    in_test_only_n = test[cat_col].apply(lambda x: x in in_test_only).value_counts().get(True, 0)
    x_in_test_only = set(X_test[cat_col]) - set(X_train[cat_col])
    x_in_test_only_n = test[cat_col].apply(lambda x: x in x_in_test_only).value_counts().get(True, 0)
    print(cat_col, in_test_only_n / test_n, x_in_test_only_n / x_test_n)


word_count = Counter()
for name in train['EntryStreetName']:
    if pd.isna(name):
        continue
    for word in name.split('_'):
        word_count[word] += 1
        
for name in train['ExitStreetName']:
    if pd.isna(name):
        continue
    for word in name.split('_'):
        word_count[word] += 1

import pdb
pdb.set_trace()
