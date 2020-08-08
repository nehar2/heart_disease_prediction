
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/svm_cross_validation.csv')
data['validation_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### COST & KERNEL ###
data_formatted = data.pivot(index='cost', columns='kernel' , values='validation_f1_score')

heatmap = sb.heatmap(data_formatted, cmap="Blues", linewidths=.5)
heatmap.invert_yaxis()
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('Kernel')
plt.ylabel('Cost')
plt.legend().set_title('Validation Test F1 Score')

plt.savefig('charts/svm_heatmap_1.png')
