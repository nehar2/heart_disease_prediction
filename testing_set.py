
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
import pickle

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train_replaced.csv')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv')
x_test = pd.read_csv('output_data/heart_disease_data_x_test_replaced.csv')
y_test = pd.read_csv('output_data/heart_disease_data_y_test.csv')

scores = {'algorithm':[], 'f1_score':[], 'precision_score':[], 'recall_score':[]}

### ADAPTIVE BOOSTING ###
clf = AdaBoostClassifier(n_estimators=17, learning_rate=0.093, base_estimator=tree.DecisionTreeClassifier(max_depth=2))
clf.fit(x_train, y_train['num'])
pickle.dump(clf, open('pickle_files/adaptive_boosting_model.pickle', 'wb'))
y_pred = clf.predict(x_test)
scores['algorithm'].append('adaptive_boosting')
scores['f1_score'].append(f1_score(y_pred, y_test['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, y_test['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, y_test['num'], average='weighted'))

### LOGISTIC REGRESSION ###
clf = LogisticRegression(solver='newton-cg')
clf.fit(x_train, y_train['num'])
pickle.dump(clf, open('pickle_files/logistic_regression_model.pickle', 'wb'))
y_pred = clf.predict(x_test)
scores['algorithm'].append('logistic_regression')
scores['f1_score'].append(f1_score(y_pred, y_test['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, y_test['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, y_test['num'], average='weighted'))

### XGBOOST ###
clf = XGBClassifier(n_estimators=14, learning_rate=0.093, max_depth=2)
clf.fit(x_train, y_train['num'])
pickle.dump(clf, open('pickle_files/xgboost_model.pickle', 'wb'))
y_pred = clf.predict(x_test)
scores['algorithm'].append('xgboost')
scores['f1_score'].append(f1_score(y_pred, y_test['num'], average='weighted'))
scores['precision_score'].append(precision_score(y_pred, y_test['num'], average='weighted'))
scores['recall_score'].append(recall_score(y_pred, y_test['num'], average='weighted'))

### SAVING SCORES ###
pd.DataFrame(scores).to_csv('charts/testing_set_scores.csv', index=None)

