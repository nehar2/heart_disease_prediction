
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
import xgboost
from xgboost import XGBClassifier
import warnings
from tqdm import tqdm
import pdb

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train.csv', index_col='patientid_train')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid_train')

cross_validation = {'n_jobs':[], 'max_depth':[], 'learning_rate':[], 'train_r2':[], 'test_r2':[]}

for n_jobs in tqdm(range (10,15)):
	for max_depth in tqdm(range(1,3)):
		for learning_rate in tqdm(range (90,100)):
			learning_rate = learning_rate/100
			xgb_model = XGBClassifier(n_jobs=n_jobs, learning_rate=learning_rate, max_depth=max_depth)
			cross_val_object = cross_validate(xgb_model, x_train, y_train['num'], cv=15, scoring=('r2'), return_train_score=True)
			cross_validation['n_jobs'].append(n_jobs)
			cross_validation['max_depth'].append(max_depth)
			cross_validation['learning_rate'].append(learning_rate)
			cross_validation['train_r2'].append(cross_val_object['train_score'].mean())
			cross_validation['test_r2'].append(cross_val_object['test_score'].mean())
			
pd.DataFrame(cross_validation).to_csv('output_cv/xgboost_cross_validation.csv', index=None)
