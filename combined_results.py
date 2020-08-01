
import pandas as pd

logistic_regression = pd.read_csv('output_cv/logistic_regression_cross_validation.csv')
logistic_regression['algorithm']='Logistic_Regression'

decision_tree = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
decision_tree['algorithm']='Decision_Tree'

random_forest = pd.read_csv('output_cv/random_forest_cross_validation.csv')
random_forest['algorithm']='Random_Forest'

adaptive_boosting = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
adaptive_boosting['algorithm']='Adaptive_Boosting'

xgboost = pd.read_csv('output_cv/xgboost_cross_validation.csv')
xgboost['algorithm']='XGBoost'

svm = pd.read_csv('output_cv/svm_cross_validation.csv')
svm['hyperparameters'] = 'cost: '+svm['cost'].astype(str) + 'kernel: '+svm['kernel'].astype(str)
svm['algorithm']='SVM'
svm_hyperparamaters = svm['hyperparameters'].mean()

combined_cross_validation_data = pd.concat([logistic_regression, svm_hyperparamaters])
combined_cross_validation_data.to_csv('output_cv/combined_results.csv', index=None)

