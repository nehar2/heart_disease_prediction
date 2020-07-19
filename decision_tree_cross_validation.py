
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.model_selection import cross_validate
import warnings
import pdb

warnings.filterwarnings('ignore')

x_train = pd.read_csv('output_data/heart_disease_data_x_train.csv', index_col='patientid_train')
y_train = pd.read_csv('output_data/heart_disease_data_y_train.csv', index_col='patientid_train')

x_train = x_train.drop('origin', axis=1)

cross_validation = {'min_samples_split':[], 'criterion':[], 'train_recall':[], 'train_precision':[], 'test_recall':[], 'test_precision':[]}

for min_samples_split in range(2,10):
    for criterion in ['mse','friedman_mse','mae']:
        clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
        cross_val_object = cross_validate(clf, x_train, y_train['num'], cv=15, scoring=('recall_weighted', 'precision_weighted'), return_train_score=True)
        cross_validation['min_samples_split'].append(min_samples_split)
        cross_validation['criterion'].append(criterion)
        cross_validation['train_recall'].append(cross_val_object['train_recall_weighted'].mean())
        cross_validation['train_precision'].append(cross_val_object['train_precision_weighted'].mean())
        cross_validation['test_recall'].append(cross_val_object['test_recall_weighted'].mean())
        cross_validation['test_precision'].append(cross_val_object['test_precision_weighted'].mean())

pd.DataFrame(cross_validation).to_csv('output_cv/decision_tree_cross_validation.csv', index=None)


