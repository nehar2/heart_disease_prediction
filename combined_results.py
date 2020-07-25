
import pandas as pd

decisiontree = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
decisiontree['algorithm']='Decision_Tree'

'''
randomforest = pd.read_csv('output_cv/random_forest_cross_validation.csv')
randomforest['algorithm']='Random_Forest'
'''

adaboost = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
adaboost['algorithm']='Adaptive_Boosting'

'''
xgboost = pd.read_csv('output_cv/extreme_gradient_boosting_cross_validation.csv')
xgboost['algorithm']='Extreme_Gradient_Boosting'
'''

svm = pd.read_csv('output_cv/svm_cross_validation.csv')
svm['algorithm']='SVM'

combined_cross_validation_data = pd.concat([decisiontree, adaboost, svm])
combined_cross_validation_data.to_csv('output_cv/combined_results.csv', index=None)

