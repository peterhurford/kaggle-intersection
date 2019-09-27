import math
import string

import pandas as pd
import numpy as np

from sklearn.neighbors import DistanceMetric

from utils import print_step


print_step('Load train')
train = pd.read_csv('train.csv')

print_step('Load test')
test = pd.read_csv('test.csv')

print_step('Concatenating')
train.loc[:, 'is_train'] = 1
test.loc[:, 'is_train'] = 0
tr_te = pd.concat([train, test], sort=False).reset_index(drop=True)

print_step('Imputing')
tr_te = tr_te.fillna('Empty')

print_step('Matches')
tr_te['StreetsMatch'] = (tr_te['EntryStreetName'] == tr_te['ExitStreetName']).astype(int)
tr_te['HeadingsMatch'] = (tr_te['EntryHeading'] == tr_te['ExitHeading']).astype(int)
tr_te['StreetsHeadingsMatch'] = (tr_te['HeadingsMatch'] & tr_te['StreetsMatch']).astype(int)

print_step('Cleanup')
tr_te['Path'] = tr_te['City'] + '_' + tr_te['Path']
tr_te['Path'] = tr_te['Path'].apply(lambda s: s.replace(' ', '_'))

tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
tr_te['EntryStreetName'] = tr_te['City'] + '_' + tr_te['EntryStreetName']
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace(' ', '_'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_St', '_Street'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_Ave', '_Avenue'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_Bld', '_Boulevard'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_Pkwy', '_Parkway'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('S_', 'South_'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_S\b', '_South'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('N_', 'North'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_N\b', '_North'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('E_', 'East'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_E\b', '_East'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('W_', 'West'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('_W\b', '_West'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('NW', 'Northwest'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('NE', 'Northeast'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('SW', 'Southwest'))
tr_te['EntryStreetName'] = tr_te['EntryStreetName'].apply(lambda s: s.replace('SE', 'Southeast'))

tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
tr_te['ExitStreetName'] = tr_te['City'] + '_' + tr_te['ExitStreetName']
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace(' ', '_'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace(' ', '_'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_St', '_Street'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_Ave', '_Avenue'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_Bld', '_Boulevard'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_Pkwy', '_Parkway'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('S_', 'South_'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_S\b', '_South'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('N_', 'North'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_N\b', '_North'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('E_', 'East'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_E\b', '_East'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('W_', 'West'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('_W\b', '_West'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('NW', 'Northwest'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('NE', 'Northeast'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('SW', 'Southwest'))
tr_te['ExitStreetName'] = tr_te['ExitStreetName'].apply(lambda s: s.replace('SE', 'Southeast'))

tr_te['IntersectionId'] = tr_te['City'] + '_' + tr_te['IntersectionId'].astype(str)


print_step('Street Type')
def street_type(street):
    street_types = ['Street', 'Avenue', 'Road', 'Boulevard', 'Drive', 'Park', 'Parkway',
                    'Place', 'Circle', 'Highway', 'Way', 'Square', 'Terrace',
                    'Connector', 'Bridge', 'Overpass']
    for street_type in street_types:
        if '_{}'.format(street_type) in street:
            return street_type
    return 'Other'

tr_te['EntryStreetType'] = tr_te['EntryStreetName'].apply(street_type)
tr_te['ExitStreetType'] = tr_te['ExitStreetName'].apply(street_type)


print_step('Hour Bin')
def hour_bin(hour):
	if hour in [8, 9, 14, 15, 16, 17, 18]:
		return 'Rush'
	if hour in [0, 1, 2, 3, 4, 5, 6]:
		return 'Early'
	if hour in [7, 10, 11, 12, 13]:
		return 'Near rush'
	if hour in [19, 20, 21, 22, 23]:
		return 'Late'
def weekend_hour_bin(hour):
	if hour in [3, 4, 5, 6, 7, 8]:
		return 'Early Weekend'
	if hour in [9, 10]:
		return 'Weekend Morning'
	if hour in [11, 12]:
		return 'Weekend Late Morning'
	if hour in [13, 14, 15, 16, 17, 18, 19]:
		return 'Weekend Afternoon'
	if hour in [20, 21, 22, 23, 0, 1, 2]:
		return 'Late Weekend'
tr_te.loc[tr_te['Weekend'] == 0, 'HourBin'] = tr_te['Hour'].apply(hour_bin)
tr_te.loc[tr_te['Weekend'] == 1, 'HourBin'] = tr_te['Hour'].apply(weekend_hour_bin)


print_step('Interactions')
tr_te['WeekendHour'] = tr_te['Weekend'].astype(str) + '_' + tr_te['Hour'].astype(str)
tr_te['CityHour'] = tr_te['City'] + '_' + tr_te['Hour'].astype(str)
tr_te['CityMonth'] = tr_te['City'] + '_' + tr_te['Month'].astype(str)
tr_te['CityHourBin'] = tr_te['City'] + '_' + tr_te['HourBin'].astype(str)
tr_te['CityWeekendHour'] = tr_te['City'] + '_' + tr_te['Weekend'].astype(str) + '_' + tr_te['Hour'].astype(str)
tr_te['sin_Hour'] = np.sin(2 * np.pi * tr_te['Hour'] / 24)
tr_te['cos_Hour'] = np.cos(2 * np.pi * tr_te['Hour'] / 24)
tr_te['EntryStreetNameHeading'] = tr_te['EntryStreetName'] + '_' + tr_te['EntryHeading']
tr_te['ExitStreetNameHeading'] = tr_te['ExitStreetName'] + '_' + tr_te['ExitHeading']
tr_te['EntryStreetExitStreet'] = tr_te['EntryStreetName'] + '_' + tr_te['ExitStreetName']
tr_te['EntryStreetTypeExitStreetType'] = tr_te['EntryStreetType'] + '_' + tr_te['ExitStreetType']
tr_te['EntryStreetTypeHeading'] = tr_te['EntryStreetType'] + '_' + tr_te['EntryHeading']
tr_te['ExitStreetTypeHeading'] = tr_te['ExitStreetType'] + '_' + tr_te['ExitHeading']
tr_te['PathType'] = tr_te['EntryStreetTypeHeading'] + '_' +tr_te['ExitStreetTypeHeading']
tr_te['PathIntersection'] = tr_te['Path'] + '_' + tr_te['IntersectionId']
tr_te['EntryExitHeading'] = tr_te['EntryHeading'] + '_' + tr_te['ExitHeading']


print_step('Turns')
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
tr_te['EntryStreetNameTurn'] = tr_te['EntryStreetName'] + '_' + tr_te['TurnType']
tr_te['ExitStreetNameTurn'] = tr_te['ExitStreetName'] + '_' + tr_te['TurnType']
tr_te['PathNameTurn'] = tr_te['EntryStreetNameTurn'] + '_' +tr_te['ExitStreetName']
tr_te['EntryStreetTypeTurn'] = tr_te['EntryStreetType'] + '_' + tr_te['TurnType']
tr_te['ExitStreetTypeTurn'] = tr_te['ExitStreetType'] + '_' + tr_te['TurnType']
tr_te['PathTypeTurn'] = tr_te['EntryStreetTypeTurn'] + '_' +tr_te['ExitStreetType']


print_steps('Rainfall')
# Adapted from https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm
monthly_rainfall = {'Atlanta_1': 5.02, 'Atlanta_5': 3.95, 'Atlanta_6': 3.63, 'Atlanta_7': 5.12,
                    'Atlanta_8': 3.67, 'Atlanta_9': 4.09, 'Atlanta_10': 3.11, 'Atlanta_11': 4.10,
				    'Atlanta_12': 3.82, 'Boston_1': 3.92, 'Boston_5': 3.24, 'Boston_6': 3.22,
                    'Boston_7': 3.06, 'Boston_8': 3.37, 'Boston_9': 3.47, 'Boston_10': 3.79,
				    'Boston_11': 3.98, 'Boston_12': 3.73, 'Chicago_1': 1.75, 'Chicago_5': 3.38,
                    'Chicago_6': 3.63, 'Chicago_7': 3.51, 'Chicago_8': 4.62, 'Chicago_9': 3.27,
                    'Chicago_10': 2.71, 'Chicago_11': 3.01, 'Chicago_12': 2.43,
                    'Philadelphia_1': 3.52, 'Philadelphia_5': 3.88, 'Philadelphia_6': 3.29,
                    'Philadelphia_7': 4.39, 'Philadelphia_8': 3.82, 'Philadelphia_9':3.88,
                    'Philadelphia_10': 2.75, 'Philadelphia_11': 3.16, 'Philadelphia_12': 3.31}
tr_te['monthly_rainfall'] = tr_te['CityMonth'].map(monthly_rainfall)


print_step('City centers')
city_centers = {'Chicago': {'lat': 41.8781, 'lon': -87.6298},
                'Philadelphia': {'lat': 39.9526, 'lon': -75.1652},
                'Atlanta': {'lat': 33.7490, 'lon': -84.3880},
                'Boston': {'lat': 42.3601, 'lon': -71.0589}}
distance_metrics = {'haversine': DistanceMetric.get_metric('haversine'),
                    'manhattan': DistanceMetric.get_metric('manhattan'),
                    'chebyshev': DistanceMetric.get_metric('chebyshev'),
                    'euclidean': DistanceMetric.get_metric('euclidean')}
for city, center in city_centers.items():
    print_step('... {}'.format(city))
    tr_te.loc[tr_te['City'] == city, 'lat_from_center'] = tr_te.loc[tr_te['City'] == city]['Latitude'] - center['lat']
    tr_te.loc[tr_te['City'] == city, 'lon_from_center'] = tr_te.loc[tr_te['City'] == city]['Longitude'] - center['lon']
    for metric_name, metric_object in distance_metrics.items():
        print_step('... ... {}'.format(metric_name))
        tr_te.loc[tr_te['City'] == city, '{}_from_center'.format(metric_name)] = tr_te[tr_te['City'] == city].apply(lambda x: metric_object.pairwise([[x['Latitude'], x['Longitude']], [center['lat'], center['lon']]])[0][1], axis=1)
    print_step('... ... Angle')
    tr_te.loc[tr_te['City'] == city, 'angle_from_center'] = tr_te[tr_te['City'] == city].apply(lambda x: math.degrees(math.atan2(x['Latitude'] - center['lat'], x['Longitude'] - center['lon'])), axis=1)


print_step('Airports')
city_airports = {'Chicago': {'lat': 41.9742, 'lon': -87.9073},
                 'Philadelphia': {'lat': 39.8744, 'lon': -75.2424},
                 'Atlanta': {'lat': 33.6407, 'lon': -84.4277},
                 'Boston': {'lat': 42.3656, 'lon': -71.0096}}
for city, airport in city_airports.items():
    print_step('... {} (airport)'.format(city))
    tr_te.loc[tr_te['City'] == city, 'lat_from_airport'] = tr_te.loc[tr_te['City'] == city]['Latitude'] - airport['lat']
    tr_te.loc[tr_te['City'] == city, 'lon_from_airport'] = tr_te.loc[tr_te['City'] == city]['Longitude'] - airport['lon']
    for metric_name, metric_object in distance_metrics.items():
        print_step('... ... {}'.format(metric_name))
        tr_te.loc[tr_te['City'] == city, '{}_from_airport'.format(metric_name)] = tr_te[tr_te['City'] == city].apply(lambda x: metric_object.pairwise([[x['Latitude'], x['Longitude']], [airport['lat'], airport['lon']]])[0][1], axis=1)
    print_step('... ... Angle')
    tr_te.loc[tr_te['City'] == city, 'angle_from_airport'] = tr_te[tr_te['City'] == city].apply(lambda x: math.degrees(math.atan2(x['Latitude'] - airport['lat'], x['Longitude'] - airport['lon'])), axis=1)


print_step('Quadrants')
tr_te['abs_lat_from_center'] = tr_te['lat_from_center'].apply(lambda x: np.abs(x))
tr_te['abs_lon_from_center'] = tr_te['lon_from_center'].apply(lambda x: np.abs(x))
tr_te['abs_lat_from_airport'] = tr_te['lat_from_airport'].apply(lambda x: np.abs(x))
tr_te['abs_lon_from_airport'] = tr_te['lon_from_airport'].apply(lambda x: np.abs(x))
tr_te['north_south'] = tr_te['lat_from_center'].apply(lambda x: x >= 0).astype(int)
tr_te['east_west'] = tr_te['lon_from_center'].apply(lambda x: x >= 0).astype(int)
tr_te['quadrant'] = tr_te['north_south'].astype(str) + '_' + tr_te['east_west'].astype(str)
tr_te['City_north_south'] = tr_te['City'] + '_' + tr_te['north_south'].astype(str)
tr_te['City_east_west'] = tr_te['City'] + '_' + tr_te['east_west'].astype(str)
tr_te['City_quadrant'] = tr_te['City'] + '_' + tr_te['quadrant']


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
