
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
import xgboost
from xgboost import XGBClassifier
import warnings
from tqdm import tqdm
import pdb

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train_replaced.csv', index_col='patientid')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid')

cross_validation = {'n_estimators':[], 'max_depth':[], 'learning_rate':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for n_estimators in tqdm(range (2,20)):
	for max_depth in tqdm(range(1,3)):
		for learning_rate in tqdm(range (90,100)):
			learning_rate = learning_rate/1000
			xgb_model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
			cross_val_object = cross_validate(xgb_model, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
			cross_validation['n_estimators'].append(n_estimators)
			cross_validation['max_depth'].append(max_depth)
			cross_validation['learning_rate'].append(learning_rate)
			cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
			cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
			cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
			cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())
			
pd.DataFrame(cross_validation).to_csv('output_cv/xgboost_cross_validation.csv', index=None)
