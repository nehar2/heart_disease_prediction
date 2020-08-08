
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/random_forest_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### CRITERION = ENTROPY, BOOSTRAP = TRUE, MIN_SAMPLES_SPLIT & N_ESTIMATOR ###
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('Min Samples Split')
plt.ylabel('N Estimators')
plt.legend().set_title('Validation Test F1 Score')

fig, ((ax, ax), (ax, ax)) = plt.subplots(2, 2)

data_1 = data.loc[(data['criterion']=='entropy') & (data['bootstrap']==True), :]
data_1_formatted = data_1.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')
heatmap_1 = sb.heatmap(data_1_formatted, cmap="Blues", linewidths=.5)

data_2 = data.loc[(data['criterion']=='entropy') & (data['bootstrap']==False), :]
data_2_formatted = data_2.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')
heatmap_2 = sb.heatmap(data_2_formatted, cmap="Blues", linewidths=.5)

data_3 = data.loc[(data['criterion']=='gini') & (data['bootstrap']==True), :]
data_3_formatted = data_3.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')
heatmap_3 = sb.heatmap(data_3_formatted, cmap="Blues", linewidths=.5)

data_4 = data.loc[(data['criterion']=='gini') & (data['bootstrap']==False), :]
data_4_formatted = data_4.pivot(index='n_estimators', columns='min_samples_split' , values='validation_test_f1_score')
heatmap_4 = sb.heatmap(data_4_formatted, cmap="Blues", linewidths=.5)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

plt.savefig('charts/random_forest_heatmap_1.png')
