import pandas as pd
import numpy as np

from utils import print_step_step

print_step('Load train')
train = pd.read_csv('train.csv')

print_step('Load test')
test = pd.read_csv('test.csv')

print_step('Munging train')
train = train.fillna('Empty')
train['HeadingsMatch'] = (train['EntryHeading'] == train['ExitHeading'])
train['StreetsMatch'] = (train['EntryStreetName'] == train['ExitStreetName'])
train['StreetsHeadingsMatch'] = (train['HeadingsMatch'] & train['StreetsMatch'])
train['Path'] = train['City'] + '_' + train['Path']
train['Path'] = train['Path'].apply(lambda s: s.replace(' ', '_'))
train['EntryStreetName'] = train['City'] + '_' + train['EntryStreetName']
train['EntryStreetName'] = train['EntryStreetName'].apply(lambda s: s.replace(' ', '_'))
train['ExitStreetName'] = train['City'] + '_' + train['ExitStreetName']
train['ExitStreetName'] = train['ExitStreetName'].apply(lambda s: s.replace(' ', '_'))
train['WeekendHour'] = train['Weekend'].astype(str) + '_' + train['Hour'].astype(str)
train['CityHour'] = train['City'] + '_' + train['Hour'].astype(str)
train['CityWeekendHour'] = train['City'] + '_' + train['Weekend'].astype(str) + '_' + train['Hour'].astype(str)
train['sin_Hour'] = np.sin(2 * np.pi * train['Hour'] / 24)
train['cos_Hour'] = np.cos(2 * np.pi * train['Hour'] / 24)
train['EntryStreetNameHeading'] = train['EntryStreetName'] + '_' + train['EntryHeading']
train['ExitStreetNameHeading'] = train['ExitStreetName'] + '_' + train['ExitHeading']
train['EntryStreetExitStreet'] = train['EntryStreetName'] + '_' + train['ExitStreetName']
train['IntersectionId'] = train['City'] + '_' + train['IntersectionId'].astype(str)

print_step('Munging test')
test = test.fillna('Empty')
test['HeadingsMatch'] = (test['EntryHeading'] == test['ExitHeading'])
test['StreetsMatch'] = (test['EntryStreetName'] == test['ExitStreetName'])
test['StreetsHeadingsMatch'] = (test['HeadingsMatch'] & test['StreetsMatch'])
test['Path'] = test['City'] + '_' + test['Path']
test['Path'] = test['Path'].apply(lambda s: s.replace(' ', '_'))
test['EntryStreetName'] = test['City'] + '_' + test['EntryStreetName']
test['EntryStreetName'] = test['EntryStreetName'].apply(lambda s: s.replace(' ', '_'))
test['ExitStreetName'] = test['City'] + '_' + test['ExitStreetName']
test['ExitStreetName'] = test['ExitStreetName'].apply(lambda s: s.replace(' ', '_'))
test['WeekendHour'] = test['Weekend'].astype(str) + '_' + test['Hour'].astype(str)
test['CityHour'] = test['City'] + '_' + test['Hour'].astype(str)
test['CityWeekendHour'] = test['City'] + '_' + test['Weekend'].astype(str) + '_' + test['Hour'].astype(str)
test['sin_Hour'] = np.sin(2 * np.pi * test['Hour'] / 24)
test['cos_Hour'] = np.cos(2 * np.pi * test['Hour'] / 24)
test['EntryStreetNameHeading'] = test['EntryStreetName'] + '_' + test['EntryHeading']
test['ExitStreetNameHeading'] = test['ExitStreetName'] + '_' + test['ExitHeading']
test['EntryStreetExitStreet'] = test['EntryStreetName'] + '_' + test['ExitStreetName']
test['IntersectionId'] = test['City'] + '_' + test['IntersectionId'].astype(str)

print_step('Saving train')
train.to_csv('processed_train.csv', index=False)

print_step('Saving test')
test.to_csv('processed_test.csv', index=False)
