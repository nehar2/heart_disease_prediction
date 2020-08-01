
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('output_cv/xgboost_cross_validation.csv')

data_formatted = data.pivot(index='__', columns='__' , values='test_r2')

heatmap = sb.heatmap(data_formatted)
plt.title('Heatmap of Logistic Regression Performance')
plt.xlabel('__')
plt.ylabel('__')
plt.legend().set_title('Test R2 Score')

plt.savefig('charts/xgboost_heatmap.png')