import math
import string

import numpy as np
import pandas as pd

from multiprocessing import Pool

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import LabelEncoder

from utils import print_step


print_step('Load train')
train = pd.read_csv('train.csv')

print_step('Load test')
test = pd.read_csv('test.csv')

print_step('Concatenating')
train.loc[:, 'is_train'] = 1
test.loc[:, 'is_train'] = 0
tr_te = pd.concat([train, test], sort=False).reset_index(drop=True)


def feature_engineer(df):
    print_step('Imputing')
    df = df.fillna('Empty')

    print_step('Matches')
    df['StreetsMatch'] = (df['EntryStreetName'] == df['ExitStreetName']).astype(int)
    df['HeadingsMatch'] = (df['EntryHeading'] == df['ExitHeading']).astype(int)
    df['StreetsHeadingsMatch'] = (df['HeadingsMatch'] & df['StreetsMatch']).astype(int)

    print_step('Cleanup')

    print_step('...Path/Intersection')
# https://www.kaggle.com/c/bigquery-geotab-intersection-congestion/discussion/108770
    df['IntersectionId'] = df['City'] + '_' + df['IntersectionId'].astype(str)
    df['Path'] = df['City'] + '_' + df['Path']
    df['Path'] = df['Path'].apply(lambda s: s.replace(' ', '_'))

    print_step('...Hour')
    df['HourCat'] = df['Hour'].apply(lambda h: 'H{}'.format(h))
    print_step('...Month')
    df['MonthCat'] = df['Month'].apply(lambda h: 'M{}'.format(h))

    find_replaces = [[' ', '_'], ['_St', '_Street'], ['_Ave', '_Avenue'], ['_Bld', '_Boulevard'],
                     ['_Pkway', '_Parkway'], ['S_', 'South_'], ['_S\b', '_South'], ['N_', 'North'],
                     ['_N\b', '_North'], ['_N\b', '_North'], ['E_', 'East'], ['_E\b', '_East'],
                     ['W_', 'West'], ['_W\b', '_West'], ['NW', 'Northwest'], ['NE', 'Northeast'],
                     ['SW', 'Southwest'], ['SE', 'Southeast']]
    i = 1
    total = len(find_replaces) * 2
    for street in ['EntryStreetName', 'ExitStreetName']:
        df[street] = df[street].apply(lambda s: s.translate(str.maketrans('', '', string.punctuation)))
        df[street] = df['City'] + '_' + df[street]
        for find_replace in find_replaces:
            find = find_replace[0]
            replace = find_replace[1]
            print_step('...({}/{}) {} {} -> {}'.format(i, total, street, find, replace))
            df[street] = df[street].apply(lambda s: s.replace(find, replace))
            i += 1


    print_step('Street Type')
    def street_type(street):
        street_types = ['Street', 'Avenue', 'Road', 'Boulevard', 'Drive', 'Park', 'Parkway',
                        'Place', 'Circle', 'Highway', 'Way', 'Square', 'Terrace',
                        'Connector', 'Bridge', 'Overpass']
        for street_type in street_types:
            if '_{}'.format(street_type) in street:
                return street_type
        return 'Other'

    df['EntryStreetType'] = df['EntryStreetName'].apply(street_type)
    df['ExitStreetType'] = df['ExitStreetName'].apply(street_type)

    def has_cardinal_direction(street):
        for direction in ['North', 'South', 'West', 'East']:
            if direction in street:
                return True
        return False
    df['EntryStreetHasCardinalDirection'] = df['EntryStreetName'].apply(has_cardinal_direction).astype(int)
    df['ExitStreetHasCardinalDirection'] = df['ExitStreetName'].apply(has_cardinal_direction).astype(int)

    df['EntryStreetHasNumber'] = df['EntryStreetName'].apply(lambda s: int(any(c.isdigit() for c in s)))
    df['ExitStreetHasNumber'] = df['ExitStreetName'].apply(lambda s: int(any(c.isdigit() for c in s)))


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
    df.loc[df['Weekend'] == 0, 'HourBin'] = df['Hour'].apply(hour_bin)
    df.loc[df['Weekend'] == 1, 'HourBin'] = df['Hour'].apply(weekend_hour_bin)


    print_step('Interactions')
    interactions = {'IntersectionId': ['EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'Path', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'EntryStreetName': ['ExitStreetName', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'ExitStreetName': ['EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'EntryStreetType': ['ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'ExitStreetType': ['EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'EntryHeading': ['ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'ExitHeading': ['HourCat', 'MonthCat', 'HourBin', 'Weekend'],
                    'HourCat': ['MonthCat', 'Weekend'],
                    'MonthCat': ['HourBin'],
                    'City': ['MonthCat', 'HourCat', 'HourBin', 'Weekend', 'EntryHeading', 'ExitHeading', 'EntryStreetType', 'ExitStreetType']}
    def interact(df, interactions):
        i = 1
        total = sum([len(list(set(v))) for k, v in interactions.items()])
        for var_a, vars_b in interactions.items():
            for var_b in list(set(vars_b)):
                label = '{}_X_{}'.format(var_a, var_b)
                print_step('...({}/{}) {}'.format(i, total, label))
                df[label] = df[var_a].astype(str) + '_' + df[var_b].astype(str)
                i += 1
        return df
    df = interact(df, interactions)
    interactions = {'City': ['HourCat_X_MonthCat', 'MonthCat_X_HourBin', 'HourCat_X_Weekend', 'EntryStreetType_X_ExitStreetType', 'EntryHeading_X_ExitHeading',
                             'ExitStreetType_X_ExitHeading', 'EntryStreetType_X_EntryHeading'],
                    'IntersectionId': ['EntryStreetName_X_ExitStreetName', 'EntryHeading_X_ExitHeading', 'HourCat_X_Weekend'],
                    'HourBin': ['EntryStreetType_X_ExitStreetType', 'EntryHeading_X_ExitHeading', 'EntryStreetType_X_EntryHeading', 'ExitStreetType_X_ExitHeading'],
                    'HourCat': ['EntryStreetType_X_ExitStreetType', 'EntryHeading_X_ExitHeading', 'EntryStreetType_X_EntryHeading', 'ExitStreetType_X_ExitHeading'],
                    'MonthCat': ['HourCat_X_Weekend']}
    df = interact(df, interactions)
    interactions = {'City': ['MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                             'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading'],
                   'MonthCat': ['HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                                'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading',
                                'HourCat_X_EntryStreetType_X_ExitStreetType', 'HourCat_X_EntryHeading_X_ExitHeading',
                                'HourCat_X_EntryStreetType_X_EntryHeading', 'HourCat_X_ExitStreetType_X_ExitHeading']}
    df = interact(df, interactions)
                    

    print_step('Turns')
    turns = {'Straight': ['E_E', 'N_N', 'S_S', 'W_W', 'NE_NE', 'SE_SE', 'NW_NW', 'SW_SW'],
             'Slight Left': ['E_NE', 'NE_N', 'N_NW', 'NW_W', 'W_SW', 'SW_S', 'S_SE', 'SE_E'],
             'Left': ['E_N', 'N_W', 'W_S', 'S_E', 'NE_NW', 'NW_SW', 'SE_NE', 'SW_SE'],
             'Sharp Left': ['E_NW', 'N_SW', 'W_SE', 'S_NE', 'NE_W', 'NW_S', 'SE_N', 'SW_E'],
             'Slight Right': ['E_SE', 'N_NE', 'W_NW', 'S_SW', 'NE_E', 'NW_N', 'SE_S', 'SW_W'],
             'Right': ['E_S', 'N_E', 'W_N', 'S_W', 'NE_SE', 'NW_NE', 'SE_SW', 'SW_NW'],
             'Sharp Right': ['E_SW', 'N_SE', 'W_NE', 'S_NW', 'NE_S', 'NW_E', 'SE_W', 'SW_N'],
             'UTurn': ['E_W', 'N_S', 'W_E', 'S_N', 'NE_SW', 'SE_NW', 'NW_SE', 'SW_NE']}
    i = 1
    total = sum([len(v) for k, v in turns.items()])
    turns2 = {}
    for turn_type, turn_set in turns.items():
        for turn in turn_set:
            print_step('... ({}/{}) {} {}'.format(i, total, turn_type, turn))
            turns2[turn] = turn_type
            i += 1
    df['TurnType'] = df['EntryHeading_X_ExitHeading'].apply(lambda t: turns2[t])

    turns3 = {'Straight': 'Straight', 'Slight Left': 'Slight Turn', 'Left': 'Turn',
              'Sharp Left': 'Sharp Turn', 'Slight Right': 'Slight Turn', 'Right': 'Turn',
              'Sharp Right': 'Sharp Turn', 'UTurn': 'UTurn'}
    df['TurnSharpness'] = df['TurnType'].apply(lambda t: turns3[t])

    turns4 = {'Straight': 'Straight', 'Slight Left': 'Left', 'Left': 'Left',
              'Sharp Left': 'Left', 'Slight Right': 'Right', 'Right': 'Right',
              'Sharp Right': 'Right', 'UTurn': 'UTurn'}
    df['TurnDirection'] = df['TurnType'].apply(lambda t: turns4[t])


    print_step('City centers')
    city_centers = {'Chicago': {'lat': 41.8781, 'lon': -87.6298},
                    'Philadelphia': {'lat': 39.9526, 'lon': -75.1652},
                    'Atlanta': {'lat': 33.7490, 'lon': -84.3880},
                    'Boston': {'lat': 42.3601, 'lon': -71.0589}}
    distance_metrics = {'haversine': DistanceMetric.get_metric('haversine'),
                        'manhattan': DistanceMetric.get_metric('manhattan'),
                        'chebyshev': DistanceMetric.get_metric('chebyshev'),
                        'euclidean': DistanceMetric.get_metric('euclidean')}
    i = 1
    for city, center in city_centers.items():
        print_step('... ({}/{}) {}'.format(i, 4, city))
        i += 1
        df.loc[df['City'] == city, 'lat_from_center'] = df.loc[df['City'] == city]['Latitude'] - center['lat']
        df.loc[df['City'] == city, 'lon_from_center'] = df.loc[df['City'] == city]['Longitude'] - center['lon']
        j = 1
        total = len(distance_metrics.keys()) + 1
        for metric_name, metric_object in distance_metrics.items():
            print_step('... ... ({}/{}) {}'.format(j, total, metric_name))
            df.loc[df['City'] == city, '{}_from_center'.format(metric_name)] = df[df['City'] == city].apply(lambda x: metric_object.pairwise([[x['Latitude'], x['Longitude']], [center['lat'], center['lon']]])[0][1], axis=1)
            j += 1
        print_step('... ... ({}/{}) Angle'.format(total, total))
        df.loc[df['City'] == city, 'angle_from_center'] = df[df['City'] == city].apply(lambda x: math.degrees(math.atan2(x['Latitude'] - center['lat'], x['Longitude'] - center['lon'])), axis=1)


    print_step('Airports')
    city_airports = {'Chicago': {'lat': 41.9742, 'lon': -87.9073},
                     'Philadelphia': {'lat': 39.8744, 'lon': -75.2424},
                     'Atlanta': {'lat': 33.6407, 'lon': -84.4277},
                     'Boston': {'lat': 42.3656, 'lon': -71.0096}}
    i = 1
    for city, airport in city_airports.items():
        print_step('... ({}/{}) {} - airport'.format(i, 4, city))
        i += 1
        df.loc[df['City'] == city, 'lat_from_airport'] = df.loc[df['City'] == city]['Latitude'] - airport['lat']
        df.loc[df['City'] == city, 'lon_from_airport'] = df.loc[df['City'] == city]['Longitude'] - airport['lon']
        j = 1
        total = len(distance_metrics.keys()) + 1
        for metric_name, metric_object in distance_metrics.items():
            print_step('... ... ({}/{}) {}'.format(j, total, metric_name))
            df.loc[df['City'] == city, '{}_from_airport'.format(metric_name)] = df[df['City'] == city].apply(lambda x: metric_object.pairwise([[x['Latitude'], x['Longitude']], [airport['lat'], airport['lon']]])[0][1], axis=1)
        print_step('... ... ({}/{}) Angle'.format(total, total))
        df.loc[df['City'] == city, 'angle_from_airport'] = df[df['City'] == city].apply(lambda x: math.degrees(math.atan2(x['Latitude'] - airport['lat'], x['Longitude'] - airport['lon'])), axis=1)


    print_step('Quadrants')
    df['abs_lat_from_center'] = df['lat_from_center'].apply(lambda x: np.abs(x))
    df['abs_lon_from_center'] = df['lon_from_center'].apply(lambda x: np.abs(x))
    df['abs_lat_from_airport'] = df['lat_from_airport'].apply(lambda x: np.abs(x))
    df['abs_lon_from_airport'] = df['lon_from_airport'].apply(lambda x: np.abs(x))
    df['NorthSouth'] = df['lat_from_center'].apply(lambda x: x >= 0).astype(int)
    df['EastWest'] = df['lon_from_center'].apply(lambda x: x >= 0).astype(int)


    print_step('Interactions II')
    interactions = {'TurnType': ['IntersectionId', 'EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading',
                                 'HourCat', 'MonthCat', 'HourBin', 'Weekend', 'City'],
                    'TurnSharpness': ['IntersectionId', 'EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading',
                                      'HourCat', 'MonthCat', 'HourBin', 'Weekend', 'City'],
                    'TurnDirection': ['IntersectionId', 'EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading',
                                      'HourCat', 'MonthCat', 'HourBin', 'Weekend', 'City'],
                    'NorthSouth': ['City', 'EastWest', 'EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin',
                                   'Weekend'],
                    'EastWest': ['City', 'EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend']}
    df = interact(df, interactions)
    interactions = {'EntryStreetType': ['TurnType_X_ExitStreetType', 'TurnSharpness_X_ExitStreetType', 'TurnDirection_X_ExitStreetType'],
                    'City': ['TurnType_X_HourCat', 'TurnType_X_HourBin', 'TurnSharpness_X_HourCat', 'TurnSharpness_X_HourBin', 'TurnDirection_X_HourCat', 'TurnDirection_X_HourBin',
                             'NorthSouth_X_EastWest'],
                    'NorthSouth_X_EastWest': ['EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend',
                                              'TurnType_X_HourCat', 'TurnType_X_HourBin', 'TurnSharpness_X_HourCat', 'TurnSharpness_X_HourBin', 'TurnDirection_X_HourCat', 'TurnDirection_X_HourBin']}
    df = interact(df, interactions)
    interactions = {'City_X_NorthSouth_X_EastWest': ['EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend',
                                                     'TurnType', 'TurnSharpness', 'TurnDirection', 'TurnType_X_HourCat', 'TurnType_X_HourBin', 'TurnSharpness_X_HourCat', 'TurnSharpness_X_HourBin',
                                                     'TurnDirection_X_HourCat', 'TurnDirection_X_HourBin', 'MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType',
                                                     'HourBin_X_EntryHeading_X_ExitHeading', 'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading'],
                    'EastWest_X_City': ['EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'
                                        'TurnType', 'TurnSharpness', 'TurnDirection', 'TurnType_X_HourCat', 'TurnType_X_HourBin', 'TurnSharpness_X_HourCat', 'TurnSharpness_X_HourBin',
                                        'TurnDirection_X_HourCat', 'TurnDirection_X_HourBin'],
                    'NorthSouth_X_City': ['EntryStreetName', 'ExitStreetName', 'EntryStreetType', 'ExitStreetType', 'EntryHeading', 'ExitHeading', 'HourCat', 'MonthCat', 'HourBin', 'Weekend'
                                          'TurnType', 'TurnSharpness', 'TurnDirection', 'TurnType_X_HourCat', 'TurnType_X_HourBin', 'TurnSharpness_X_HourCat', 'TurnSharpness_X_HourBin',
                                          'TurnDirection_X_HourCat', 'TurnDirection_X_HourBin'],
                    'EastWest': ['MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                                 'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading'],
                    'NorthSouth': ['MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                                   'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading'],
                    'EastWest_X_City': ['MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                                 'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading'],
                    'NorthSouth_X_City': ['MonthCat_X_HourCat_X_Weekend', 'HourBin_X_EntryStreetType_X_ExitStreetType', 'HourBin_X_EntryHeading_X_ExitHeading',
                                          'HourBin_X_EntryStreetType_X_EntryHeading', 'HourBin_X_ExitStreetType_X_ExitHeading']}
    df = interact(df, interactions)


    print_step('Rainfall')
# Adapted from https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm
    monthly_rainfall = {'Atlanta_M1': 5.02, 'Atlanta_M5': 3.95, 'Atlanta_M6': 3.63, 'Atlanta_M7': 5.12,
                        'Atlanta_M8': 3.67, 'Atlanta_M9': 4.09, 'Atlanta_M10': 3.11, 'Atlanta_M11': 4.10,
                        'Atlanta_M12': 3.82, 'Boston_M1': 3.92, 'Boston_M5': 3.24, 'Boston_M6': 3.22,
                        'Boston_M7': 3.06, 'Boston_M8': 3.37, 'Boston_M9': 3.47, 'Boston_M10': 3.79,
                        'Boston_M11': 3.98, 'Boston_M12': 3.73, 'Chicago_M1': 1.75, 'Chicago_M5': 3.38,
                        'Chicago_M6': 3.63, 'Chicago_M7': 3.51, 'Chicago_M8': 4.62, 'Chicago_M9': 3.27,
                        'Chicago_M10': 2.71, 'Chicago_M11': 3.01, 'Chicago_M12': 2.43,
                        'Philadelphia_M1': 3.52, 'Philadelphia_M5': 3.88, 'Philadelphia_M6': 3.29,
                        'Philadelphia_M7': 4.39, 'Philadelphia_M8': 3.82, 'Philadelphia_M9':3.88,
                        'Philadelphia_M10': 2.75, 'Philadelphia_M11': 3.16, 'Philadelphia_M12': 3.31}
    df['monthly_rainfall'] = df['City_X_MonthCat'].map(monthly_rainfall)

    return df


def parallelize_dataframe_by_row(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def parallelize_dataframe_by_col(df, func, n_cores=4):
    df_cols = df.columns
    df_split = np.array_split(df_cols, n_cores)
    df_split = [df[cols] for cols in df_split]
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split), axis=1)
    pool.close()
    pool.join()
    return df


tr_te = parallelize_dataframe_by_row(tr_te, feature_engineer, n_cores=46)


def label_encode(df):
    print_step('Label encode')
    cat_cols = df.dtypes[(df.dtypes != np.float) & (df.dtypes != np.int64)]
    targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
               'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
               'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
               'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
               'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
               'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
               'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
    cat_cols = [c for c in cat_cols.keys() if c not in targets]

    i = 1
    total = len(cat_cols)
    for c in sorted(cat_cols):
        print_step('...({}/{}) {}'.format(i, total, c))
        df[c] = pd.Categorical(df[c]).codes
        i += 1
    return df

tr_te = parallelize_dataframe_by_col(tr_te, label_encode, n_cores=46)


def count_encode(df):
    print_step('Count encoding')
    cat_cols = df.dtypes[(df.dtypes != np.float) & (df.dtypes != np.int64)]
    targets = ['TotalTimeStopped_p20', 'TotalTimeStopped_p40',
               'TotalTimeStopped_p50', 'TotalTimeStopped_p60', 'TotalTimeStopped_p80',
               'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40',
               'TimeFromFirstStop_p50', 'TimeFromFirstStop_p60',
               'TimeFromFirstStop_p80', 'DistanceToFirstStop_p20',
               'DistanceToFirstStop_p40', 'DistanceToFirstStop_p50',
               'DistanceToFirstStop_p60', 'DistanceToFirstStop_p80']
    cat_cols = [c for c in cat_cols.keys() if c not in targets]
    for col in sorted(cat_cols):
        label = '{}__COUNT'.format(col)
        print_step('...{}'.format(label))
        df.loc[:, label] = df.groupby(col)[col].transform('count')
    return df

tr_te = parallelize_dataframe_by_col(tr_te, count_encode, n_cores=46)


print_step('Unique counts')
counts = {'IntersectionId': ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path', 'HourCat_X_Weekend', 'HourBin', 'Month', 'TurnType', 'TurnSharpness', 'TurnDirection'],
          'Path': ['IntersectionId', 'HourCat_X_Weekend', 'HourBin', 'Month'],
          'EntryStreetName': ['IntersectionId', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path', 'HourCat_X_Weekend', 'HourBin', 'Month', 'TurnType', 'TurnSharpness', 'TurnDirection'],
          'ExitStreetName': ['IntersectionId', 'EntryStreetName', 'EntryHeading', 'ExitHeading', 'Path', 'HourCat_X_Weekend', 'HourBin', 'Month', 'TurnType', 'TurnSharpness', 'TurnDirection']}
for var_a, vars_b in counts.items():
    for var_b in vars_b:
        label = '{}_NUNIQUE_{}'.format(var_a, var_b)
        print_step('...{}'.format(label))
        tr_te[label] = tr_te.groupby(var_a)[var_b].transform('nunique')

print_step('Splitting')
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


# Other helpful kernels used
# https://www.kaggle.com/jpmiller/eda-to-break-through-rmse-68
# https://www.kaggle.com/bgmello/how-one-percentile-affect-the-others
# https://www.kaggle.com/danofer/baseline-feature-engineering-geotab-69-5-lb
# https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367
