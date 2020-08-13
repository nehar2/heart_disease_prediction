
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/logistic_regression_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### L1 Ratio & SOLVER ###
data_formatted = data.pivot(index='l1_ratio', columns='solver' , values='validation_test_f1_score')

plt.figure(figsize=(4, 9))
heatmap = sb.heatmap(data_formatted, cmap="Blues")
heatmap.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('Logistic Regression Heatmap')
plt.xlabel('Solver')
plt.ylabel('L1 Ratio')
plt.savefig('charts/logistic_regression_heatmap.png')