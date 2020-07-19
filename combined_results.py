
import pandas as pd

svm = pd.read_csv('output_cv/svm_cross_validation.csv')
svm['model']='SVM'

adaboost = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
adaboost['model']='Adaptive_Boosting'

randomforest = pd.read_csv('output_cv/random_forest_cross_validation.csv')
randomforest['model']='Random_Forest'

decisiontree = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
decisiontree['model']='Decision_Tree'

combined_cross_validation_data = pd.concat([svm, adaboost, randomforest, decisiontree])
combined_cross_validation_data.to_csv('output_cv/combined_results.csv', index=None)

