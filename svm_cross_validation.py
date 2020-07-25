
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_validate
import warnings

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train.csv', index_col='patientid_train')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid_train')

cross_validation = {'kernel':[], 'cost':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for cost in range(1,5):
    cost = cost/10
    for kernel in ['poly','rbf','sigmoid']:
        clf = svm.SVC(kernel=kernel, C=cost)
        cross_val_object = cross_validate(clf, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
        cross_validation['kernel'].append(kernel)
        cross_validation['cost'].append(cost)
        cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
        cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
        cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
        cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())

pd.DataFrame(cross_validation).to_csv('output_cv/svm_cross_validation.csv', index=None)


