
import pandas as pd
import pickle
from sklearn.metrics import f1_score

x_test = pd.read_csv('output_data/heart_disease_data_x_test.csv')
x_test_replaced = pd.read_csv('output_data/heart_disease_data_x_test_replaced.csv')
y_test = pd.read_csv('output_data/heart_disease_data_y_test.csv')

x_test_replaced = x_test_replaced.set_index('patientid')
x_test_replaced = x_test_replaced.reindex(index=x_test['patientid'])
x_test_replaced = x_test_replaced.reset_index()

y_test = y_test.set_index('patientid')
y_test = y_test.reindex(index=x_test['patientid'])
y_test = y_test.reset_index()

clf = pickle.load(open('pickle_files/logistic_regression_model.pickle', 'rb'))
y_pred = clf.predict(x_test_replaced)
data = pd.DataFrame({'y_prediction': y_pred, 'y_actual': y_test['num'], 'location': x_test['origin']})
info = {'location':[], 'f1_score':[], 'size':[]}

for location in data['location'].unique():
	data_location = data.loc[data['location'] == location]
	location_f1_score = f1_score(data_location['y_prediction'], data_location['y_actual'], average='weighted')
	info['location'].append(location)
	info['f1_score'].append(location_f1_score)
	info['size'].append(data_location.shape[0])

pd.DataFrame(info).to_csv('charts/logistic_regression_model_performance.csv', index=None)