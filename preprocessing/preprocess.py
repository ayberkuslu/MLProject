import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


loc = r"C:\Users\me\Documents\datasets\pump.csv"
loc2 = r"C:\Users\me\Documents\datasets\pump_labels.csv"

# Load data.
df = pd.read_csv(loc)
labels = pd.read_csv(loc2)

# Merge data and labels together in one dataframe.
df = pd.merge(df, labels, on='id')
del labels


df.apply(lambda x: sum(x.isnull()))


df.funder.value_counts()



# Create a function to reduce the amount of dummy columns needed whilst maintaining the 
# information contained in the column.

def funder_wrangler(row):  
    '''Keep top 5 values and set the rest to 'other'''

    if row['funder']=='Government Of Tanzania':
        return 'gov'
    elif row['funder']=='Danida':
        return 'danida'
    elif row['funder']=='Hesawa':
        return 'hesawa'
    elif row['funder']=='Rwssp':
        return 'rwssp'
    elif row['funder']=='World Bank':
        return 'world_bank'    
    else:
        return 'other'
    
df['funder'] = df.apply(lambda row: funder_wrangler(row), axis=1)




# Add column named 'status_group_vals' to allow the use of a pivot table to check differences
# between the different funders.

vals_to_replace = {'functional':2, 'functional needs repair':1,
                   'non functional':0}

df['status_group_vals']  = df.status_group.replace(vals_to_replace)


piv_table = pd.pivot_table(df,index=['funder','status_group'],
                           values='status_group_vals', aggfunc='count')
piv_table



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

print('Percent functional danida: ', round(percent_functional_danida,3))
print('Percent functional gov: ', round(percent_functional_gov,3))
print('Percent functional hesawa: ', round(percent_functional_hesawa,3))
print('Percent functional non gov: ', round(percent_functional_non_gov,3))
print('Percent functional rwssp: ', round(percent_functional_rwssp,3))
print('Percent functional world bank: ', round(percent_functional_world_bank,3))



df.installer.value_counts()



def installer_wrangler(row):
    '''Keep top 5 values and set the rest to 'other'''
    if row['installer']=='DWE':
        return 'dwe'
    elif row['installer']=='Government':
        return 'gov'
    elif row['installer']=='RWE':
        return 'rwe'
    elif row['installer']=='Commu':
        return 'commu'
    elif row['installer']=='DANIDA':
        return 'danida'
    else:
        return 'other'  

df['installer'] = df.apply(lambda row: installer_wrangler(row), axis=1)