
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train_replaced.csv', index_col='patientid')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid')

x_train = x_train.astype('category')

cross_validation = {'min_samples_split':[], 'n_estimators':[], 'criterion':[], 'bootstrap':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for min_samples_split in tqdm(range(5,30)):
	for n_estimators in tqdm(range(50,150)):
		for criterion in tqdm(['entropy', 'gini']):
			for bootstrap in tqdm([True, False]):
				clf = RandomForestClassifier(min_samples_split=min_samples_split, n_estimators=n_estimators, criterion=criterion, bootstrap=bootstrap)
				cross_val_object = cross_validate(clf, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
				cross_validation['min_samples_split'].append(min_samples_split)
				cross_validation['n_estimators'].append(n_estimators)
				cross_validation['criterion'].append(criterion)
				cross_validation['bootstrap'].append(bootstrap)
				cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
				cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
				cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
				cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())

pd.DataFrame(cross_validation).to_csv('output_cv/random_forest_cross_validation.csv', index=None)
