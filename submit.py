import pandas as pd
import numpy as np

from utils import print_step, rmse

print_step('Load train')
train = pd.read_csv('train.csv')

print_step('Gathering')
oof_data = {}
submit_data = {}
targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
           'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
           'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
           'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
           'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
           'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
           'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
for target in targets:
    if 'TimeFromFirstStop' not in target and '40' not in target and '60' not in target: # These targets don't count
        oof_data[target] = pd.read_csv('{}_oof.csv'.format(target))
        submit_data[target] = pd.read_csv('{}_submit.csv'.format(target))


print_step('Compiling OOFs')
oofs = pd.concat(oof_data.values(), axis=1).reset_index(drop=True)


rmses_ = []
for target in oof_data.keys():
	oof_data[target]['target'] = train[target]
	rmse_ = rmse(oof_data[target]['target'], oof_data[target][target])
	print(target, rmse_)
	rmses_.append(rmse_)
print('Mean', np.mean(rmses_))

oofs2 = pd.concat([pd.DataFrame(x.values) for x in oof_data.values()], axis=0)
oofs2.columns = ['value', 'target']

print('Global', rmse(oofs2['target'], oofs2['value']))


print_step('Compiling Submit')
prediction = []
for x in range(len(list(submit_data.values())[0])):
    for y in list(submit_data.values()):
        prediction.append(y.values[x][0])

submission = pd.read_csv('sample_submission.csv')
submission['Target'] = prediction
import pdb
pdb.set_trace()
submission.to_csv('submission.csv', index = False)
