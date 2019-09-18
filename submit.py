import sys

import pandas as pd
import numpy as np

from utils import print_step, rmse

print_step('Load train')
train = pd.read_csv('processed_train.csv')

print_step('Load test')
test = pd.read_csv('processed_test.csv')

print_step('Gathering')
IS_OOFS_MODE = len(sys.argv) == 2 and sys.argv[1] == 'add_oofs'
oof_data = {}
oof_g_data = {}
submit_data = {}
submit_g_data = {}
targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
           'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
           'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
           'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
           'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
           'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
           'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
if IS_OOFS_MODE:
    print_step('[Adding lvl 1 OOF data]')
for target in targets:
    if 'TimeFromFirstStop' not in target and '40' not in target and '60' not in target: # These targets don't count
        label = target
        if IS_OOFS_MODE:
            label = label + '_w_oofs'
        oof_data[target] = pd.read_csv('{}_oof.csv'.format(label))
        oof_g_data[target] = pd.read_csv('{}_g_oof.csv'.format(label))
        submit_data[target] = pd.read_csv('{}_submit.csv'.format(label))
        submit_g_data[target] = pd.read_csv('{}_g_submit.csv'.format(label))


print_step('Compiling OOFs')
oofs = pd.concat(oof_data.values(), axis=1).reset_index(drop=True)
oofs_g = pd.concat(oof_g_data.values(), axis=1).reset_index(drop=True)
oofs_g.columns = [c + '_g' for c in oofs_g.columns]

rmses_ = []
for target in oof_data.keys():
    label = target
    if IS_OOFS_MODE:
        label = label + '_w_oofs'
    oof_data[target]['target'] = train[target]
    rmse_ = rmse(oof_data[target]['target'], oof_data[target][label])
    print(target, rmse_)
    rmses_.append(rmse_)
rmse_mean = np.mean(rmses_)
print('Mean', rmse_mean)

oofs3 = pd.concat([pd.DataFrame(x.values) for x in oof_data.values()], axis=0)
oofs3.columns = ['value', 'target']

global_mean = rmse(oofs3['target'], oofs3['value'])
print('Global', global_mean)

rmses_ = []
for target in oof_data.keys():
    label = target
    if IS_OOFS_MODE:
        label = label + '_w_oofs'
    oof_g_data[target]['target'] = train[target]
    rmse_ = rmse(oof_g_data[target]['target'], oof_g_data[target][label])
    print(target, rmse_)
    rmses_.append(rmse_)
group_mean = np.mean(rmses_)
print('Group Mean', group_mean)
oofs3_g = pd.concat([pd.DataFrame(x.values) for x in oof_g_data.values()], axis=0)
oofs3_g.columns = ['value', 'target']
global_mean2 = rmse(oofs3_g['target'], oofs3_g['value'])
print('Global2', global_mean2)
print('Projected LB', (global_mean / rmse_mean) * group_mean)

import pdb
pdb.set_trace()


print_step('Compiling Submit')
submit = pd.concat(submit_data.values(), axis=1).reset_index(drop=True)
submit_g = pd.concat(submit_g_data.values(), axis=1).reset_index(drop=True)
submit_g.columns = [c + '_g' for c in submit_g.columns]

submit3 = pd.concat([pd.DataFrame(x.values) for x in submit_data.values()], axis=0)
submit3.columns = ['Target']
submit3 = submit3.reset_index(drop=True)
submit3['TargetId'] = sum([['{}_{}'.format(d, c) for d in range(submit_data['TotalTimeStopped_p20'].shape[0])] for c in range(6)], [])
submit3 = submit3.sort_values('TargetId').reset_index(drop=True)[['TargetId', 'Target']]

print_step('Saving Submit')
submit3.to_csv('submission.csv', index=False)

print_step('Compiling Submit G')
submit3 = pd.concat([pd.DataFrame(x.values) for x in submit_g_data.values()], axis=0)
submit3.columns = ['Target']
submit3 = submit3.reset_index(drop=True)
submit3['TargetId'] = sum([['{}_{}'.format(d, c) for d in range(submit_data['TotalTimeStopped_p20'].shape[0])] for c in range(6)], [])
submit3 = submit3.sort_values('TargetId').reset_index(drop=True)[['TargetId', 'Target']]

print_step('Saving Submit G')
submit3.to_csv('submission_g.csv', index=False)

print_step('Compiling Submit M')
in_only = set(test['IntersectionId']) - set(train['IntersectionId'])
test['IntersectionIdIn'] = test['IntersectionId'].apply(lambda x: x in in_only)

submit_g = pd.concat(submit_g_data.values(), axis=1).reset_index(drop=True)
for target in oof_data.keys():
    label = target
    if IS_OOFS_MODE:
        label = label + '_w_oofs'
    test.loc[test['IntersectionIdIn'] == True, target] = submit_g[test['IntersectionIdIn'] == True][label]
    test.loc[test['IntersectionIdIn'] == False, target] = submit[test['IntersectionIdIn'] == False][label]
    submit_data[target] = test[target]

submit3 = pd.concat([pd.DataFrame(x.values) for x in submit_data.values()], axis=0)
submit3.columns = ['Target']
submit3 = submit3.reset_index(drop=True)
submit3['TargetId'] = sum([['{}_{}'.format(d, c) for d in range(submit_data['TotalTimeStopped_p20'].shape[0])] for c in range(6)], [])
submit3 = submit3.sort_values('TargetId').reset_index(drop=True)[['TargetId', 'Target']]

print_step('Saving Submit M')
submit3.to_csv('submission_m.csv', index=False)

if not IS_OOFS_MODE:
    print_step('Saving OOFs (train) 1/2')
    oofs.to_csv('oofs_train.csv', index=False)
    print_step('Saving OOFs (train) 2/2')
    oofs_g.to_csv('oofs_g_train.csv', index=False)

    print_step('Saving OOFs (test) 1/2')
    submit.to_csv('oofs_test.csv', index=False)
    print_step('Saving OOFs (test) 2/2')
    submit_g.to_csv('oofs_g_test.csv', index=False)
