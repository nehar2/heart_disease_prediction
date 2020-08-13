
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/decision_tree_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### MIN_SAMPLES_SPLIT & CRITERION ###
data_formatted = data.pivot(index='min_samples_split', columns='criterion' , values='validation_test_f1_score')

plt.figure(figsize=(4, 5))
heatmap = sb.heatmap(data_formatted, cmap="Blues")
heatmap.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Decision Tree Heatmap')
plt.xlabel('Criterion')
plt.ylabel('Min Samples Split')
plt.savefig('charts/decision_tree_heatmap.png')