
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
data['test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

data_formatted = data.pivot(index='min_samples_split', columns='criterion' , values='test_f1_score')

heatmap = sb.heatmap(data_formatted)
plt.title('Heatmap of Decision Tree Performance')
plt.xlabel('Criterion')
plt.ylabel('Min Samples Split')
plt.legend().set_title('Test F1 Score')

plt.savefig('charts/decision_tree_heatmap.png')