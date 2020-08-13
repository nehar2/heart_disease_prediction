
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/svm_cross_validation.csv')
data['validation_test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

### COST & KERNEL ###
data_formatted = data.pivot(index='cost', columns='kernel' , values='validation_test_f1_score')

plt.figure(figsize=(4,8))
heatmap = sb.heatmap(data_formatted, cmap="Blues")
heatmap.invert_yaxis()
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.title('SVM Heatmap')
plt.xlabel('Kernel')
plt.ylabel('Cost')
plt.savefig('charts/svm_heatmap.png')
