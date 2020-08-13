
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings

warnings.filterwarnings('ignore')

train_x = pd.read_csv('output_data/heart_disease_data_x_train_replaced.csv')
train_y = pd.read_csv('output_data/heart_disease_data_y_train.csv')
test_x = pd.read_csv('output_data/heart_disease_data_x_test_replaced.csv')
test_y = pd.read_csv('output_data/heart_disease_data_y_test.csv')

scores = {'algorithm':[], 'f1_score':[], 'precision_score':[], 'recall_score':[]}

### RANDOM FOREST ### 
clf = RandomForestClassifier(min_samples_split=6, n_estimators=94, criterion='gini', bootstrap=True)
clf.fit(train_x, train_y['num'])
y_pred = clf.predict(test_x)
scores['algorithm'].append('random_forest')
scores['f1_score'].append(f1_score(y_pred, test_y['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, test_y['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, test_y['num'], average='weighted'))

### ADAPTIVE BOOSTING ###
clf = AdaBoostClassifier(n_estimators=17, learning_rate=0.09300000000000001, base_estimator=tree.DecisionTreeClassifier(max_depth=2))
clf.fit(train_x, train_y['num'])
y_pred = clf.predict(test_x)
scores['algorithm'].append('adaptive_boosting')
scores['f1_score'].append(f1_score(y_pred, test_y['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, test_y['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, test_y['num'], average='weighted'))

### XGBOOST ###
clf = XGBClassifier(n_estimators=14, learning_rate=0.09300000000000001, max_depth=2)
clf.fit(train_x, train_y['num'])
y_pred = clf.predict(test_x)
scores['algorithm'].append('xgboost')
scores['f1_score'].append(f1_score(y_pred, test_y['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, test_y['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, test_y['num'], average='weighted'))

pd.DataFrame(scores).to_csv('charts/testing_set_scores.csv', index=None)


