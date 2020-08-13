
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/random_forest_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### CRITERION = ENTROPY, BOOSTRAP = TRUE, MIN_SAMPLES_SPLIT & N_ESTIMATOR ###
data_1 = data.loc[(data['criterion']=='entropy') & (data['bootstrap']==True), :]
data_1_formatted = data_1.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')

plt.figure(figsize=(5, 7))
heatmap_1 = sb.heatmap(data_1_formatted, cmap="Blues")
heatmap_1.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Random Forest Heatmap 1')
plt.xlabel('Min Samples Split')
plt.ylabel('N Estimators')
plt.savefig('charts/random_forest_heatmap_1.png')

### CRITERION = ENTROPY, BOOSTRAP = FALSE, MIN_SAMPLES_SPLIT & N_ESTIMATOR ###
data_2 = data.loc[(data['criterion']=='entropy') & (data['bootstrap']==False), :]
data_2_formatted = data_2.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')

plt.figure(figsize=(5, 7))
heatmap_2 = sb.heatmap(data_2_formatted, cmap="Blues")
heatmap_2.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Random Forest Heatmap 2')
plt.xlabel('Min Samples Split')
plt.ylabel('N Estimators')
plt.savefig('charts/random_forest_heatmap_2.png')

### CRITERION = GINI, BOOSTRAP = TRUE, MIN_SAMPLES_SPLIT & N_ESTIMATOR ###
data_3 = data.loc[(data['criterion']=='gini') & (data['bootstrap']==True), :]
data_3_formatted = data_3.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')

plt.figure(figsize=(5, 7))
heatmap_3 = sb.heatmap(data_3_formatted, cmap="Blues")
heatmap_3.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Random Forest Heatmap 3')
plt.xlabel('Min Samples Split')
plt.ylabel('N Estimators')
plt.savefig('charts/random_forest_heatmap_3.png')

### CRITERION = GINI, BOOSTRAP = FALSE, MIN_SAMPLES_SPLIT & N_ESTIMATOR ###
data_4 = data.loc[(data['criterion']=='gini') & (data['bootstrap']==False), :]
data_4_formatted = data_4.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')

plt.figure(figsize=(5, 7))
heatmap_4 = sb.heatmap(data_4_formatted, cmap="Blues")
heatmap_4.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Random Forest Heatmap 4')
plt.xlabel('Min Samples Split')
plt.ylabel('N Estimators')
plt.savefig('charts/random_forest_heatmap_4.png')

