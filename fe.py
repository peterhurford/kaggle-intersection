import pandas as pd
import numpy as np

from utils import print_step

print_step('Load train')
train = pd.read_csv('train.csv')

print_step('Load test')
test = pd.read_csv('test.csv')

print_step('Munging')
train.loc[:, 'is_train'] = 1
test.loc[:, 'is_train'] = 0
tr_te = pd.concat([train, test], sort=False)
tr_te = tr_te.fillna('Empty')
tr_te['StreetsMatch'] = (tr_te['EntryStreetName'] == tr_te['ExitStreetName'])
tr_te['HeadingsMatch'] = (tr_te['EntryHeading'] == tr_te['ExitHeading'])
tr_te['StreetsHeadingsMatch'] = (tr_te['HeadingsMatch'] & tr_te['StreetsMatch'])
tr_te['Path'] = tr_te['City'] + '_' + tr_te['Path']
tr_te['Path'] = tr_te['Path'].apply(lambda s: s.replace(' ', '_'))
tr_te['EntryStreetName'] = tr_te['City'] + '_' + tr_te['EntryStreetName']
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace(' ', '_'))
tr_te['ExitStreetName'] = tr_te['City'] + '_' + tr_te['ExitStreetName']
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace(' ', '_'))
tr_te['WeekendHour'] = tr_te['Weekend'].astype(str) + '_' + tr_te['Hour'].astype(str)
tr_te['CityHour'] = tr_te['City'] + '_' + tr_te['Hour'].astype(str)
tr_te['CityWeekendHour'] = tr_te['City'] + '_' + tr_te['Weekend'].astype(str) + '_' + tr_te['Hour'].astype(str)
tr_te['sin_Hour'] = np.sin(2 * np.pi * tr_te['Hour'] / 24)
tr_te['cos_Hour'] = np.cos(2 * np.pi * tr_te['Hour'] / 24)
tr_te['EntryStreetNameHeading'] = tr_te['EntryStreetName'] + '_' + tr_te['EntryHeading']
tr_te['ExitStreetNameHeading'] = tr_te['ExitStreetName'] + '_' + tr_te['ExitHeading']
tr_te['EntryStreetExitStreet'] = tr_te['EntryStreetName'] + '_' + tr_te['ExitStreetName']
tr_te['IntersectionId'] = tr_te['City'] + '_' + tr_te['IntersectionId'].astype(str)
tr_te['PathIntersection'] = tr_te['Path'] + '_' + tr_te['IntersectionId']
tr_te['EntryExitHeading'] = tr_te['EntryHeading'] + '_' + tr_te['ExitHeading']

turns = {'Straight': ['E_E', 'N_N', 'S_S', 'W_W', 'NE_NE', 'SE_SE', 'NW_NW', 'SW_SW'],
         'Slight Left': ['E_NE', 'NE_N', 'N_NW', 'NW_W', 'W_SW', 'SW_S', 'S_SE', 'SE_E'],
         'Left': ['E_N', 'N_W', 'W_S', 'S_E', 'NE_NW', 'NW_SW', 'SE_NE', 'SW_SE'],
         'Sharp Left': ['E_NW', 'N_SW', 'W_SE', 'S_NE', 'NE_W', 'NW_S', 'SE_N', 'SW_E'],
         'Slight Right': ['E_SE', 'N_NE', 'W_NW', 'S_SW', 'NE_E', 'NW_N', 'SE_S', 'SW_W'],
         'Right': ['E_S', 'N_E', 'W_N', 'S_W', 'NE_SE', 'NW_NE', 'SE_SW', 'SW_NW'],
         'Sharp Right': ['E_SW', 'N_SE', 'W_NE', 'S_NW', 'NE_S', 'NW_E', 'SE_W', 'SW_N'],
		 'UTurn': ['E_W', 'N_S', 'W_E', 'S_N', 'NE_SW', 'SE_NW', 'NW_SE', 'SW_NE']}
turns2 = {}
for turn_type, turn_set in turns.items():
	for turn in turn_set:
		turns2[turn] = turn_type
tr_te['TurnType'] = tr_te['EntryExitHeading'].apply(lambda t: turns2[t])

turns3 = {'Straight': 'Straight', 'Slight Left': 'Slight Turn', 'Left': 'Turn',
          'Sharp Left': 'Sharp Turn', 'Slight Right': 'Slight Turn', 'Right': 'Turn',
          'Sharp Right': 'Sharp Turn', 'UTurn': 'UTurn'}
tr_te['TurnSharpness'] = tr_te['TurnType'].apply(lambda t: turns3[t])

turns4 = {'Straight': 'Straight', 'Slight Left': 'Left', 'Left': 'Left',
          'Sharp Left': 'Left', 'Slight Right': 'Right', 'Right': 'Right',
          'Sharp Right': 'Right', 'UTurn': 'UTurn'}
tr_te['TurnDirection'] = tr_te['TurnType'].apply(lambda t: turns4[t])


print_step('Count encoding')
cat_cols = tr_te.dtypes[(tr_te.dtypes != np.float) & (tr_te.dtypes != np.int64)]
targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
           'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
           'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
           'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
           'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
           'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
           'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
cat_cols = [c for c in cat_cols.keys() if c not in targets]
for col in cat_cols:
    tr_te.loc[:, '{}_count'.format(col)] = tr_te.groupby(col)[col].transform('count')
train = tr_te[tr_te['is_train'] == 1]
test = tr_te[tr_te['is_train'] == 0]
train = train.drop(['is_train'], axis=1)
test = test.drop(['is_train'], axis=1)
del tr_te
print(train.shape)
print(test.shape)

print_step('Saving train')
train.to_csv('processed_train.csv', index=False)

print_step('Saving test')
test.to_csv('processed_test.csv', index=False)
