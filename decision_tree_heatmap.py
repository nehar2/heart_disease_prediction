
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
data['validation_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### MIN_SAMPLES_SPLIT & CRITERION ###
data_formatted = data.pivot(index='min_samples_split', columns='criterion' , values='validation_f1_score')

heatmap = sb.heatmap(data_formatted, cmap="Blues", linewidths=.5)
heatmap.invert_yaxis()
plt.title('Heatmap of Decision Tree Performance')
plt.xlabel('Criterion')
plt.ylabel('Min Samples Split')
plt.legend().set_title('Validation Test F1 Score')

plt.savefig('charts/decision_tree_heatmap_1.png')