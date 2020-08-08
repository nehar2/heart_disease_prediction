
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('input_data/heart_disease_data_formatted.csv')
data.loc[data['num']==0, 'num'] = 'None'
data.loc[data['num']==1, 'num'] = 'Low'
data.loc[data['num']==2, 'num'] = 'Moderate'
data.loc[data['num']==3, 'num'] = 'High'
data.loc[data['num']==4, 'num'] = 'Severe'

sb.set(style='darkgrid')
sb.countplot(data=data, x='num', hue='origin', order=['None', 'Low', 'Moderate', 'High', 'Severe'])
plt.title('Heart Disease Prediction Values')
plt.xlabel('Likelihood of Presence of Heart Disease')
plt.ylabel('Number of Patients')
plt.legend().set_title('Key (Origin)')

plt.savefig('charts/prediction_values.png')
