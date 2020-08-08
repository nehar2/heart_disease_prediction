
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('output_cv/adaptive_boosting_cross_validation.csv')
data['validation_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### MAX_ITER & SOLVER ###
'''
fig = plt.figure()
data_formatted = fig.add_axes('n_estimators', 'max_depth' , 'learning_rate', projection='3d')
# data.pivot(', values='validation_f1_score', projection='3d')

heatmap = sb.heatmap(data_formatted, cmap="Blues", linewidths=.5)
heatmap.invert_yaxis()
plt.title('Heatmap of Adaptive Boosting Performance')
plt.xlabel('Max Depth')
plt.ylabel('N Estimators')
plt.legend().set_title('Validation F1 Score')

plt.savefig('charts/adaptive_boosting_heatmap_1.png')
'''

data_formatted = data.pivot(index='max_iter', columns='solver' , values='validation_f1_score')

heatmap = sb.heatmap(data_formatted, cmap="Blues", linewidths=.5)
heatmap.invert_yaxis()
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('Solver')
plt.ylabel('Max Iterations')
plt.legend().set_title('Validation F1 Score')

plt.savefig('charts/logistic_regression_heatmap_1.png')