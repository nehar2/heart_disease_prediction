
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

### CROSS VALIDATION ###

x_train = pd.read_csv('output_data/heart_disease_data_x_train.csv', index_col='patientid_train')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid_train')

cross_validation = {'max_iter':[], 'solver':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for max_iter in tqdm(range(10,15)):
    for solver in ['newton-cg','lbfgs','liblinear', 'sag', 'saga']:
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        cross_val_object = cross_validate(clf, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
        cross_validation['max_iter'].append(max_iter)
        cross_validation['solver'].append(solver)
        cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
        cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
        cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
        cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())

pd.DataFrame(cross_validation).to_csv('output_cv/logistic_regression_cross_validation.csv', index=None)


