
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

### CROSS VALIDATIONS ###

rule decision_tree_cross_validation:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/decision_tree_cross_validation.csv'
	shell: 'python decision_tree_cross_validation.py'

rule random_forest_cross_validation:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/random_forest_cross_validation.csv'
	shell: 'python random_forest_cross_validation.py'

rule adaptive_boosting_cross_validation:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/adaptive_boosting_cross_validation.csv'
	shell: 'python adaptive_boosting_cross_validation.py'

'''
rule extreme_gradient_boosting_cross_validation:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/extreme_gradient_boosting_cross_validation.csv'
	shell: 'python extreme_gradient_boosting_cross_validation.py'
'''

rule svm_cross_validation:
	input: 'output_data/heart_disease_data_x_train.csv', 'output_data/heart_disease_data_y_train.csv'
	output: 'output_cv/svm_cross_validation.csv'
	shell: 'python svm_cross_validation.py'

### RESULTS ###

rule combined_results:
	input: 
		'output_cv/decision_tree_cross_validation.csv', 'output_cv/random_forest_cross_validation.csv', 
		'output_cv/svm_cross_validation.csv', 'output_cv/adaptive_boosting_cross_validation.csv'
	output: 'output_cv/combined_results.csv'
	shell: 'python combined_results.py'

rule heatmap_cross_validation:
	input: 'output_cv/{algorithm}_cross_validation.csv'
	output: 'charts/{algorithm}_heatmap.png'
	shell: 'python heatmap_cross_validation.py --algorithm {wildcard.algorithm}'

cross_validation_algorithms = ['decision_tree', 'random_forest', 'svm', 'adaptive_boosting']

rule heatmap_generate_results:
	input: expand('charts/{algorithm}_heatmap.png', algorithm = cross_validation_algorithms)
	output: 'charts/heatmap_complete.txt'
	shell: 'echo complete > {output}'




