
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/logistic_regression_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### MAX_ITER & SOLVER ###
data_formatted = data.pivot(index='max_iter', columns='solver' , values='validation_test_f1_score')

heatmap = sb.heatmap(data_formatted, cmap="Blues", linewidths=.5)
heatmap.invert_yaxis()
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('Solver')
plt.ylabel('Max Iterations')
plt.legend().set_title('Validation Test F1 Score')

plt.savefig('charts/logistic_regression_heatmap_1.png')