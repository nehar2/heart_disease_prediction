
### DATASET ###

rule format_input_data:
	input: 'input_data/processed.cleveland.data', 'input_data/processed.hungarian.data', 'input_data/processed.switzerland.data', 'input_data/processed.va.data' 
	output: 'input_data/heart_disease_data_formatted.csv', 'charts/combined_missing_values_info.csv'
	shell: 'python format_input_data.py'

rule prediction_values:
	input: 'input_data/heart_disease_data_formatted.csv'
	output: 'charts/prediction_values.png'
	shell: 'python prediction_values.py'

rule train_test_split:
	input: 'input_data/heart_disease_data_formatted.csv'
	output: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_x_test.csv', 'output_data/heart_disease_data_y_train.csv', 'output_data/heart_disease_data_y_test.csv'
	shell: 'python train_test_split.py'

rule replace_missing_values:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_x_test.csv', 'output_data/heart_disease_data_y_train.csv', 'output_data/heart_disease_data_y_test.csv'
	output: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_x_test_replaced.csv'
	shell: 'python replace_missing_values.py'

### CROSS VALIDATIONS ###

rule logistic_regression_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/logistic_regression_cross_validation.csv'
	shell: 'python logistic_regression_cross_validation.py'

rule decision_tree_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/decision_tree_cross_validation.csv'
	shell: 'python decision_tree_cross_validation.py'

rule random_forest_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/random_forest_cross_validation.csv'
	shell: 'python random_forest_cross_validation.py'

rule adaptive_boosting_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/adaptive_boosting_cross_validation.csv'
	shell: 'python adaptive_boosting_cross_validation.py'

rule xgboost_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/xgboost_cross_validation.csv'
	shell: 'python xgboost_cross_validation.py'

rule svm_cross_validation:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/svm_cross_validation.csv'
	shell: 'python svm_cross_validation.py'

### HEATMAPS ###

rule logistic_regression_heatmap:
	input: 'output_cv/logistic_regression_cross_validation.csv'
	output: 'charts/logistic_regression_heatmap.png'
	shell: 'python logistic_regression_heatmap.py'

rule decision_tree_heatmap:
	input: 'output_cv/decision_tree_cross_validation.csv'
	output: 'charts/decision_tree_heatmap.png'
	shell: 'python decision_tree_heatmap.py'

rule random_forest_heatmap:
	input: 'output_cv/random_forest_cross_validation.csv'
	output: 'charts/random_forest_heatmap_1.png', 'charts/random_forest_heatmap_2.png', 'charts/random_forest_heatmap_3.png', 'charts/random_forest_heatmap_4.png'
	shell: 'python random_forest_heatmap.py'

rule adaptive_boosting_heatmap:
	input: 'output_cv/adaptive_boosting_cross_validation.csv'
	output: 'charts/adaptive_boosting_heatmap_1.png', 'charts/adaptive_boosting_heatmap_2.png'
	shell: 'python adaptive_boosting_heatmap.py'

rule xgboost_heatmap:
	input: 'output_cv/xgboost_cross_validation.csv'
	output: 'charts/xgboost_heatmap_1.png', 'charts/xgboost_heatmap_2.png'
	shell: 'python xgboost_heatmap.py'

rule svm_heatmap:
	input: 'output_cv/svm_cross_validation.csv'
	output: 'charts/svm_heatmap.png'
	shell: 'python svm_heatmap.py'

### RESULTS ###

rule combined_cross_validation_results:
	input: 
		'output_cv/logistic_regression_cross_validation.csv', 'output_cv/decision_tree_cross_validation.csv', 
		'output_cv/random_forest_cross_validation.csv', 'output_cv/adaptive_boosting_cross_validation.csv', 
		'output_cv/xgboost_cross_validation.csv', 'output_cv/svm_cross_validation.csv'
	output: 'charts/combined_cross_validation_results.csv'
	shell: 'python combined_cross_validation_results.py'

rule testing_set:
	input: 'output_data/heart_disease_data_x_train_replaced.csv', 'output_data/heart_disease_data_y_train.csv', 'output_data/heart_disease_data_x_test_replaced.csv', 'output_data/heart_disease_data_y_test.csv'
	output: 'charts/testing_set_scores.csv'
	shell: 'python testing_set.py'
	



