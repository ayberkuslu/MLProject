import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


loc = r"C:\Users\me\Documents\datasets\pump.csv"
loc2 = r"C:\Users\me\Documents\datasets\pump_labels.csv"

# Load data.
df = pd.read_csv(loc)
labels = pd.read_csv(loc2)

# Merge data and labels together in one dataframe.
df = pd.merge(df, labels, on='id')
del labels
df.info()

# df.head(3)
# df.shape
# Check for nulls.

df.apply(lambda x: sum(x.isnull()))

df.funder.value_counts()
# Add column named 'status_group_vals' to allow the use of a pivot table to check differences
# between the different funders.

vals_to_replace = {'functional': 2, 'functional needs repair': 1,
                   'non functional': 0}

df['status_group_vals'] = df.status_group.replace(vals_to_replace)
piv_table = pd.pivot_table(df, index=['funder', 'status_group'],
                           values='status_group_vals', aggfunc='count')
# piv_table
total_danida = piv_table[0] + piv_table[1] + piv_table[2]
percent_functional_danida = (piv_table[0] / total_danida) * 100

total_gov = piv_table[3] + piv_table[4] + piv_table[5]
percent_functional_gov = (piv_table[3] / total_gov) * 100

total_hesawa = piv_table[6] + piv_table[7] + piv_table[8]
percent_functional_hesawa = (piv_table[6] / total_hesawa) * 100

total_other = piv_table[9] + piv_table[10] + piv_table[11]
percent_functional_non_gov = (piv_table[9] / total_other) * 100

total_rwssp = piv_table[12] + piv_table[13] + piv_table[14]
percent_functional_rwssp = (piv_table[12] / total_rwssp) * 100

total_world_bank = piv_table[15] + piv_table[16] + piv_table[17]
percent_functional_world_bank = (piv_table[15] / total_world_bank) * 100

print('Percent functional danida: ', round(percent_functional_danida, 3))
print('Percent functional gov: ', round(percent_functional_gov, 3))
print('Percent functional hesawa: ', round(percent_functional_hesawa, 3))
print('Percent functional non gov: ', round(percent_functional_non_gov, 3))
print('Percent functional rwssp: ', round(percent_functional_rwssp, 3))
print('Percent functional world bank: ', round(percent_functional_world_bank, 3))
# There are some clear differences here that will hopefully improve the model. The next feature
# to inspect is 'installer'.

df.installer.value_counts()


# Create a function to reduce the amount of dummy columns needed whilst maintaining the
# information contained in the column.

def installer_wrangler(row):
    '''Keep top 5 values and set the rest to 'other'''
    if row['installer'] == 'DWE':
        return 'dwe'
    elif row['installer'] == 'Government':
        return 'gov'
    elif row['installer'] == 'RWE':
        return 'rwe'
    elif row['installer'] == 'Commu':
        return 'commu'
    elif row['installer'] == 'DANIDA':
        return 'danida'
    else:
        return 'other'


df['installer'] = df.apply(lambda row: installer_wrangler(row), axis=1)

piv_table = pd.pivot_table(df, index=['funder', 'status_group'],
                           values='status_group_vals', aggfunc='count')
# piv_table

total_dwe = piv_table[0] + piv_table[1] + piv_table[2]
percent_functional_dwe = (piv_table[0] / total_dwe) * 100

total_gov = piv_table[3] + piv_table[4] + piv_table[5]
percent_functional_gov = (piv_table[3] / total_gov) * 100

total_hesawa = piv_table[6] + piv_table[7] + piv_table[8]
percent_functional_hesawa = (piv_table[6] / total_hesawa) * 100

total_other = piv_table[9] + piv_table[10] + piv_table[11]
percent_functional_non_gov = (piv_table[9] / total_other) * 100

total_rwssp = piv_table[12] + piv_table[13] + piv_table[14]
percent_functional_rwssp = (piv_table[12] / total_rwssp) * 100

total_world_bank = piv_table[15] + piv_table[16] + piv_table[17]
percent_functional_world_bank = (piv_table[15] / total_world_bank) * 100

print('Percent functional dwe: ', round(percent_functional_dwe, 3))
print('Percent functional gov: ', round(percent_functional_gov, 3))
print('Percent functional hesawa: ', round(percent_functional_hesawa, 3))
print('Percent functional non gov: ', round(percent_functional_non_gov, 3))
print('Percent functional rwssp: ', round(percent_functional_rwssp, 3))
print('Percent functional world bank: ', round(percent_functional_world_bank, 3))

# As was the case with 'funder' there are some clear differences here that will hopefully
# improve the model. The next feature to inspect is 'subvillage'.

print(df.subvillage.value_counts())
print('Number of villages: ', len(df.subvillage.value_counts()))
# funder. It's probably not worth creating dummy variables for the top 5. I'll drop this one but
# feel free to experiment here.

df = df.drop('subvillage', axis=1)

# Let's investigate the next column containg null data: 'public_meeting'.

df.public_meeting.value_counts()
# We only have two values here: true and false. This one can stay but we'll have to replace
# the unknown data with a string value.

df.public_meeting = df.public_meeting.fillna('Unknown')

# 'scheme_management' is up next.

df.scheme_management.value_counts()


# Create a function to reduce the amount of dummy columns needed whilst maintaining the
# information contained in the column.

def scheme_wrangler(row):
    '''Keep top 5 values and set the rest to 'other'. '''
    if row['scheme_management'] == 'VWC':
        return 'vwc'
    elif row['scheme_management'] == 'WUG':
        return 'wug'
    elif row['scheme_management'] == 'Water authority':
        return 'wtr_auth'
    elif row['scheme_management'] == 'WUA':
        return 'wua'
    elif row['scheme_management'] == 'Water Board':
        return 'wtr_brd'
    else:
        return 'other'


df['scheme_management'] = df.apply(lambda row: scheme_wrangler(row), axis=1)
piv_table = pd.pivot_table(df, index=['scheme_management', 'status_group'],
                           values='status_group_vals', aggfunc='count')
# piv_table
total_other = piv_table[0] + piv_table[1] + piv_table[2]
percent_functional_other = (piv_table[0] / total_other) * 100

total_vwc = piv_table[3] + piv_table[4] + piv_table[5]
percent_functional_vwc = (piv_table[3] / total_vwc) * 100

total_wtr_auth = piv_table[6] + piv_table[7] + piv_table[8]
percent_functional_wtr_auth = (piv_table[6] / total_wtr_auth) * 100

total_wtr_brd = piv_table[9] + piv_table[10] + piv_table[11]
percent_functional_wtr_brd = (piv_table[9] / total_wtr_brd) * 100

total_wua = piv_table[12] + piv_table[13] + piv_table[14]
percent_functional_wua = (piv_table[12] / total_wua) * 100

total_wug = piv_table[15] + piv_table[16] + piv_table[17]
percent_functional_wug = (piv_table[15] / total_wug) * 100

print('Percent functional other: ', round(percent_functional_other, 3))
print('Percent functional vwc: ', round(percent_functional_vwc, 3))
print('Percent functional water authority: ', round(percent_functional_wtr_auth, 3))
print('Percent functional water board: ', round(percent_functional_wtr_brd, 3))
print('Percent functional wua: ', round(percent_functional_wua, 3))
print('Percent functional wug: ', round(percent_functional_wug, 3))
df.scheme_name.value_counts()

len(df.scheme_name.unique())

# Lots of factors and the top 5 or so only represent a fraction of the total values. Probably
# safe to drop this column.

df = df.drop('scheme_name', axis=1)
# The final column containing nulls is 'permit'.

df.permit.value_counts()
# We only have two values here: true and false. This one can stay but we'll have to replace
# the unknown data with a string value.

df.permit = df.permit.fillna('Unknown')
df.apply(lambda x: sum(x.isnull()))
# Excellent! Now there are no nulls in the data set. We can move on to look at columns with
# string values and modify or remove them as we see fit.

str_cols = df.select_dtypes(include=['object'])
str_cols.apply(lambda x: len(x.unique()))
# 'Date recorded'

df.date_recorded.describe()
# Let's first convert the column to type datetime. Then change the column to represent the
# number of days since the most recently recorded datum we have. The idea being that more
# recently recorded pumps might be more likely to be functional than non-functional.

df.date_recorded = pd.to_datetime(df.date_recorded)
df.date_recorded.describe()
# The most recent data is 2013-12-03. Subtract each date from this point to obtain a
# 'days_since_recorded' column.

df.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(df.date_recorded)
df.columns = ['days_since_recorded' if x == 'date_recorded' else x for x in df.columns]
df.days_since_recorded = df.days_since_recorded.astype('timedelta64[D]').astype(int)
df.days_since_recorded.describe()
# There's a wide range of data here hopefully it will help improve the predictive power of our
# models. Next up for inspection is 'wpt_name' (Name of the waterpoint if there is one).

df.wpt_name.value_counts()
# Due to the huge number of factors and the lack of a clear dominating value I'll drop this.
# I may come back and include the top 5 later. Next up is 'basin'.

df = df.drop('wpt_name', axis=1)
df.basin.value_counts()
piv_table = pd.pivot_table(df, index=['basin', 'status_group'],
                           values=['status_group_vals'], aggfunc='count')
piv_table
df.region.value_counts()
# Both basin, lga, ward and region contain geographical information so there is a risk of them being
# highly correlated with each other. I'll drop then for now.
# They could be be worth including though, so I may come back to them.
df = df.drop(['region', 'lga', 'ward'], axis=1)
df.recorded_by.value_counts()
# All data points have the same value so this offers no information that would help build our
# model.
df = df.drop('recorded_by', axis=1)
# extraction_type, extraction_type_group and extraction_type_class appear to contain very similar
# data. I'll drop the first two and keep the last one.

df = df.drop(['extraction_type', 'extraction_type_group'], axis=1)
df.management.value_counts()
# This appears to be almost identical to 'scheme_management'. I'll drop it.

df = df.drop('management', axis=1)
df.management_group.value_counts()
# Appears to offer no new info and is likely to overlap with 'scheme_management'.

df = df.drop('management_group', axis=1)
df.payment.value_counts()
df.payment_type.value_counts()
# Payment and payment_type contain identical data. Remove one and keep the other.

df = df.drop('payment', 1)
df.water_quality.value_counts()
df.quality_group.value_counts()
# Water_quality and quality_group contain identical data. Remove one and keep the other.

df = df.drop('quality_group', 1)
df.quantity.value_counts()
df.quantity_group.value_counts()
# Quantity and quantity_group contain identical data. Remove one and keep the other.

df = df.drop('quantity_group', 1)
df.source.value_counts()
# Quantity and quantity_group contain identical data. Remove one and keep the other.

df = df.drop('quantity_group', 1)
df.source.value_counts()
df.source_class.value_counts()
df.source_type.value_counts()
# Source and source_type contain very similar information. Remove one and keep the other.

df = df.drop('source', 1)
# gps_height, longitude, latitude, region_code and district_code are all geographic info which
# is unlikely to add any predictive power to the model given that there are other variables
# containing geographic data. 'num_private' hasn't been given a discription on Driven Data,
# it appears to be superflous. We expect id to not contain any useful information so that gets
# dropped too.

df = df.drop(['gps_height', 'longitude', 'latitude', 'region_code', 'district_code',
              'num_private', 'id'], axis=1)
str_cols.apply(lambda x: len(x.unique()))
df.construction_year.value_counts()


# Turn construction_year into a categorical column containing the following values: '60s', '70s',
# '80s', '90s, '00s', '10s', 'unknown'.

def construction_wrangler(row):
    if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
        return '60s'
    elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
        return '70s'
    elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
        return '80s'
    elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
        return '90s'
    elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
        return '00s'
    elif row['construction_year'] >= 2010:
        return '10s'
    else:
        return 'unknown'


df['construction_year'] = df.apply(lambda row: construction_wrangler(row), axis=1)
# sns.distplot(df.population, bins=40)
plt.show()

df.info()

# Most wells have a few hundred people living around them. There are some wells
# serving huge populations. This may skew the data.
# sns.distplot(df.amount_tsh, bins=40)
plt.show()

# This plot measures the amount of water available at the pump. It looks a lot like the
# population graph which makes sense.

df.population.describe()
df.amount_tsh.describe()

# There appears to be enough variation between the two to warrant keeping them in the model.
# Let's save the dataframe to a new csv file. We'll start creating models in the next notebooks.
df = df.drop('status_group_vals', 1)
df.to_csv('pump_train_for_models.csv', index=False)
# We'll also need to perform the same modifications to the test set.

test = pd.read_csv(r"C:\Users\me\Documents\datasets\pump_test.csv")

test = test.drop(['gps_height', 'longitude', 'latitude', 'region_code', 'district_code',
                  'num_private', 'id', 'payment', 'management_group', 'management',
                  'extraction_type', 'extraction_type_group', 'recorded_by', 'region', 'lga',
                  'ward', 'wpt_name', 'scheme_name', 'subvillage', 'quantity_group',
                  'quality_group', 'source'], axis=1)

test.date_recorded = pd.datetime(2013, 12, 3) - pd.to_datetime(test.date_recorded)
test.columns = ['days_since_recorded' if x == 'date_recorded' else x for x in test.columns]
test.days_since_recorded = test.days_since_recorded.astype('timedelta64[D]').astype(int)

test.permit = test.permit.fillna('Unknown')
test.public_meeting = test.public_meeting.fillna('Unknown')

test['scheme_management'] = test.apply(lambda row: scheme_wrangler(row), axis=1)
test['construction_year'] = test.apply(lambda row: construction_wrangler(row), axis=1)
test['installer'] = test.apply(lambda row: installer_wrangler(row), axis=1)
# test['funder'] = test.apply(lambda row: funder_wrangler(row), axis=1)
# df.shape
#
# test.shape
# The train and test sets match up. We can save the test set now.

test.to_csv('pump_test_for_models.csv', index=False)