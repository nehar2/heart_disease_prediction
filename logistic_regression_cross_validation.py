
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

### CROSS VALIDATION ###

x_train = pd.read_csv('output_data/heart_disease_data_x_train_replaced.csv', index_col='patientid')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid')

cross_validation = {'l1_ratio':[], 'solver':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for l1_ratio in tqdm(range(0,101)):
    l1_ratio = l1_ratio/100
    for solver in tqdm(['newton-cg','lbfgs','liblinear', 'sag', 'saga']):
        clf = LogisticRegression(l1_ratio=l1_ratio, solver=solver)
        cross_val_object = cross_validate(clf, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
        cross_validation['l1_ratio'].append(l1_ratio)
        cross_validation['solver'].append(solver)
        cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
        cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
        cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
        cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())

pd.DataFrame(cross_validation).to_csv('output_cv/logistic_regression_cross_validation.csv', index=None)

