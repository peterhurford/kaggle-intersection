import datarobot as dr
import pandas as pd
import numpy as np

train = pd.read_csv('processed_train.csv')
train_id = train['RowId']
train.drop('RowId', axis=1, inplace=True)

test = pd.read_csv('processed_test.csv')
test_id = test['RowId']
test.drop('RowId', axis=1, inplace=True)

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


for target in target_data.keys():
	print('{}... Creating...'.format(target))
	train['target'] = target_data[target]
	partition = dr.partitioning_methods.RandomCV(holdout_pct=0, reps=5, seed=42)
	project = dr.Project.start(train,
							   project_name='Intersection-{}'.format(target),
							   target='target',
							   metric='RMSE',
							   autopilot_on=False,
							   partitioning_method=partition,
							   worker_count=-1)
	print('{}... Modeling...'.format(target))
	kerases = [bp for bp in project.get_blueprints() if 'Keras' in bp.model_type]
	tfs = [bp for bp in project.get_blueprints() if 'TensorFlow' in bp.model_type]
	xgbs = [bp for bp in project.get_blueprints() if 'eXtreme' in bp.model_type]
	bps = sum([kerases, tfs, xgbs], [])
	jobs = [project.train(bp, sample_pct=80, scoring_type='validation') for bp in bps]
    
import pdb
pdb.set_trace()
