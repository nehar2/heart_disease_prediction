
import pandas as pd
import statistics
from statistics import median

# cleveland['origin']='Cleveland'

### CLEVELAND DATA ###
cleveland = pd.read_csv('input_data/processed.cleveland.data', header=None, na_values='?')
cleveland['origin']='Cleveland'

cleveland_missing_values = {'column':[], 'total_missing':[], 'percent_missing':[]}
for column in range(0,14):
    total_missing = cleveland[column].isnull().sum()
    percent_missing = cleveland[column].isnull().sum()/len(cleveland[column])*100
    cleveland_missing_values['column'].append(column)
    cleveland_missing_values['total_missing'].append(total_missing)
    cleveland_missing_values['percent_missing'].append(percent_missing)
cleveland_missing_values_table = pd.DataFrame(cleveland_missing_values)
cleveland_missing_values_table['origin']='cleveland'

### HUNGARY DATA ###
hungary = pd.read_csv('input_data/processed.hungarian.data', header=None, na_values='?')
hungary['origin']='Hungary'

hungary_missing_values = {'column':[], 'total_missing':[], 'percent_missing':[]}
for column in range(0,14):
    total_missing = hungary[column].isnull().sum()
    percent_missing = hungary[column].isnull().sum()/len(hungary[column])*100
    hungary_missing_values['column'].append(column)
    hungary_missing_values['total_missing'].append(total_missing)
    hungary_missing_values['percent_missing'].append(percent_missing)
hungary_missing_values_table = pd.DataFrame(hungary_missing_values)
hungary_missing_values_table['origin']='hungary'

### SWITZERLAND DATA ###
switzerland = pd.read_csv('input_data/processed.switzerland.data', header=None, na_values='?')
switzerland['origin']='Switzerland'

switzerland_missing_values = {'column':[], 'total_missing':[], 'percent_missing':[]}
for column in range(0,14):
    total_missing = switzerland[column].isnull().sum()
    percent_missing = switzerland[column].isnull().sum()/len(switzerland[column])*100
    switzerland_missing_values['column'].append(column)
    switzerland_missing_values['total_missing'].append(total_missing)
    switzerland_missing_values['percent_missing'].append(percent_missing)
switzerland_missing_values_table = pd.DataFrame(switzerland_missing_values)
switzerland_missing_values_table['origin']='switzerland'

### VIRGINIA DATA ###
virginia = pd.read_csv('input_data/processed.va.data', header=None, na_values='?')
virginia['origin']='Virginia'

virginia_missing_values = {'column':[], 'total_missing':[], 'percent_missing':[]}
for column in range(0,14):
    total_missing = virginia[column].isnull().sum()
    percent_missing = virginia[column].isnull().sum()/len(virginia[column])*100
    virginia_missing_values['column'].append(column)
    virginia_missing_values['total_missing'].append(total_missing)
    virginia_missing_values['percent_missing'].append(percent_missing)
virginia_missing_values_table = pd.DataFrame(virginia_missing_values)
virginia_missing_values_table['origin']='virginia'

### COMBINING DATASETS ###
combined_missing_values_info = pd.concat([cleveland_missing_values_table, hungary_missing_values_table, switzerland_missing_values_table, virginia_missing_values_table])
combined_missing_values_info.to_csv('charts/combined_missing_values_info.csv', index=None)

combined_heart_disease_data = pd.concat([cleveland, hungary, switzerland, virginia])
combined_heart_disease_data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'gbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num', 'origin']

### DROPPING COLUMNS BASED ON MISSING VALUES ###
combined_heart_disease_data = combined_heart_disease_data.drop(['slope', 'ca'], axis=1)

### EXPORT ###
combined_heart_disease_data.to_csv('input_data/heart_disease_data_formatted.csv', index=None)

