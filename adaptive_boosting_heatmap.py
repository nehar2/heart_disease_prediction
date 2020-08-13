
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### MAX_DEPTH = 1, LEARNING_RATE & N_ESTIMATORS ###
data_1 = data.loc[(data['max_depth']==1), :]
data_1_formatted = data_1.pivot(index='n_estimators', columns='learning_rate' , values='validation_test_f1_score')

plt.figure(figsize=(5, 5))
heatmap_1 = sb.heatmap(data_1_formatted, cmap="Blues", xticklabels=data_1_formatted.columns.values.round(3))
heatmap_1.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Adaptive Boosting Heatmap 1')
plt.xlabel('N Estimators')
plt.ylabel('Learning Rate')
plt.savefig('charts/adaptive_boosting_heatmap_1.png')

### MAX_DEPTH = 2, LEARNING_RATE & N_ESTIMATORS ###
data_2 = data.loc[(data['max_depth']==2), :]
data_2_formatted = data_2.pivot(index='n_estimators', columns='learning_rate' , values='validation_test_f1_score')

plt.figure(figsize=(5, 5))
heatmap_2 = sb.heatmap(data_2_formatted, cmap="Blues", xticklabels=data_2_formatted.columns.values.round(3))
heatmap_2.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Adaptive Boosting Heatmap 2')
plt.xlabel('N Estimators')
plt.ylabel('Learning Rate')
plt.savefig('charts/adaptive_boosting_heatmap_2.png')

