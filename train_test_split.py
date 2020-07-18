
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

data = pd.read_csv('input_data/heart_disease_data_formatted.csv')
data_features = data.loc[:, data.columns != 'num']

x_train, x_test, y_train, y_test = train_test_split(data_features, data['num'], test_size = 0.2, random_state=0)
x_train.to_csv('output_data/heart_disease_data_x_train.csv', index_label='patientid_train')
x_test.to_csv('output_data/heart_disease_data_x_test.csv', index_label='patientid_test')
y_train.to_csv('output_data/heart_disease_data_y_train.csv', index_label='patientid_train')
y_test.to_csv('output_data/heart_disease_data_y_test.csv', index_label='patientid_test')
