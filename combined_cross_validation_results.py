
import pandas as pd

logistic_regression = pd.read_csv('output_cv/logistic_regression_cross_validation.csv')
logistic_regression['algorithm']='Logistic_Regression'
logistic_regression['hyperparameters'] = 'l1_ratio: '+logistic_regression['l1_ratio'].astype(str) + ' solver: '+logistic_regression['solver'].astype(str)
logistic_regression_dropped = logistic_regression.drop(['l1_ratio', 'solver'], axis=1)

decision_tree = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
decision_tree['algorithm']='Decision_Tree'
decision_tree['hyperparameters'] = 'min_samples_split: '+decision_tree['min_samples_split'].astype(str) + ' criterion: '+decision_tree['criterion'].astype(str)
decision_tree_dropped = decision_tree.drop(['min_samples_split', 'criterion'], axis=1)

random_forest = pd.read_csv('output_cv/random_forest_cross_validation.csv')
random_forest['algorithm']='Random_Forest'
random_forest['hyperparameters'] = 'min_samples_split: '+random_forest['min_samples_split'].astype(str) + ' n_estimators: '+random_forest['n_estimators'].astype(str) + ' criterion: '+random_forest['criterion'].astype(str) + ' bootstrap: '+random_forest['bootstrap'].astype(str)
random_forest_dropped = random_forest.drop(['min_samples_split', 'n_estimators', 'criterion', 'bootstrap'], axis=1)

adaptive_boosting = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
adaptive_boosting['algorithm']='Adaptive_Boosting'
adaptive_boosting['hyperparameters'] = 'n_estimators: '+adaptive_boosting['n_estimators'].astype(str) + ' max_depth: '+adaptive_boosting['max_depth'].astype(str) + ' learning_rate: '+adaptive_boosting['learning_rate'].astype(str)
adaptive_boosting_dropped = adaptive_boosting.drop(['n_estimators', 'max_depth', 'learning_rate'], axis=1)

xgboost = pd.read_csv('output_cv/xgboost_cross_validation.csv')
xgboost['algorithm']='XGBoost'
xgboost['hyperparameters'] = 'n_estimators: '+xgboost['n_estimators'].astype(str) + ' max_depth: '+xgboost['max_depth'].astype(str) + ' learning_rate: '+xgboost['learning_rate'].astype(str)
xgboost_dropped = xgboost.drop(['n_estimators', 'max_depth', 'learning_rate'], axis=1)

svm = pd.read_csv('output_cv/svm_cross_validation.csv')
svm['algorithm']='SVM'
svm['hyperparameters'] = 'cost: '+svm['cost'].astype(str) + ' kernel: '+svm['kernel'].astype(str)
svm_dropped = svm.drop(['kernel', 'cost'], axis=1)

combined_cross_validation_data = pd.concat([logistic_regression_dropped, decision_tree_dropped, random_forest_dropped, adaptive_boosting_dropped, xgboost_dropped, svm_dropped])

combined_cross_validation_data['validation_test_f1_score'] = 2*((combined_cross_validation_data['test_recall']*combined_cross_validation_data['test_precision'])/(combined_cross_validation_data['test_recall']+combined_cross_validation_data['test_precision']))
combined_cross_validation_data['validation_train_f1_score'] = 2*((combined_cross_validation_data['train_recall']*combined_cross_validation_data['train_precision'])/(combined_cross_validation_data['train_recall']+combined_cross_validation_data['train_precision']))

combined_cross_validation_data.to_csv('charts/combined_cross_validation_results.csv', index=None)

