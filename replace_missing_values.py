
import pandas as pd
import sklearn
import statistics
from statistics import median

x_train = pd.read_csv('output_data/heart_disease_data_x_train.csv')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv')
x_test = pd.read_csv('output_data/heart_disease_data_x_test.csv')
y_test = pd.read_csv('output_data/heart_disease_data_y_test.csv')

### X_TRAIN DATASET ###

cleveland_train = x_train.loc[x_train['origin'] == 'Cleveland']
cleveland_train_replaced = cleveland_train.fillna(cleveland_train.median())

hungary_train = x_train.loc[x_train['origin'] == 'Hungary']
hungary_train_replaced = hungary_train.fillna(hungary_train.median())

switzerland_train = x_train.loc[x_train['origin'] == 'Switzerland']
switzerland_train_replaced = switzerland_train.fillna(switzerland_train.median())

virginia_train = x_train.loc[x_train['origin'] == 'Virginia']
virginia_train_replaced = virginia_train.fillna(virginia_train.median())

x_train_replaced = pd.concat([cleveland_train_replaced, hungary_train_replaced, switzerland_train_replaced, virginia_train_replaced])

### REARRANGE BASED ON PATIENTID VALUES ###

x_train_replaced = x_train_replaced.set_index('patientid')
x_train_replaced = x_train_replaced.reindex(index=y_train['patientid'])
x_train_replaced = x_train_replaced.reset_index()

### DROP ORIGIN ###
x_train_replaced = x_train_replaced.drop(['origin'], axis=1)

x_train_replaced.to_csv('output_data/heart_disease_data_x_train_replaced.csv', index=None)

### X_TEST DATASET ###

cleveland_test = x_test.loc[x_test['origin'] == 'Cleveland']
cleveland_test_replaced = cleveland_test.fillna(cleveland_test.median())

hungary_test = x_test.loc[x_test['origin'] == 'Hungary']
hungary_test_replaced = hungary_test.fillna(hungary_test.median())

switzerland_test = x_test.loc[x_test['origin'] == 'Switzerland']
switzerland_test_replaced = switzerland_test.fillna(switzerland_test.median())

virginia_test = x_test.loc[x_test['origin'] == 'Virginia']
virginia_test_replaced = virginia_test.fillna(virginia_test.median())

x_test_replaced = pd.concat([cleveland_test_replaced, hungary_test_replaced, switzerland_test_replaced, virginia_test_replaced])

### REARRANGE BASED ON PATIENTID VALUES ###

x_test_replaced = x_test_replaced.set_index('patientid')
x_test_replaced = x_test_replaced.reindex(index=y_test['patientid'])
x_test_replaced = x_test_replaced.reset_index()

### DROP ORIGIN ###
x_test_replaced = x_test_replaced.drop(['origin'], axis=1)

x_test_replaced.to_csv('output_data/heart_disease_data_x_test_replaced.csv', index=None)

