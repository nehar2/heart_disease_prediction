
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/logistic_regression_cross_validation.csv')
data['test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

data_formatted = data.pivot(index='max_iter', columns='solver' , values='test_f1_score')

heatmap = sb.heatmap(data_formatted)
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('Solver')
plt.ylabel('Max Iterations')
plt.legend().set_title('Test F1 Score')

plt.savefig('charts/logistic_regression_heatmap.png')