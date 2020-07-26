
import argparse
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, help='algorithm', default=None)
args = parser.parse_args()
print(args)

data = pd.read_csv('output_cv/{algorithm}_cross_validation.csv'.format(algorithm=args.algorithm))
data['test_f1_score'] = 2*((data['test_recall']*data['test_precision'])/(data['test_recall']+data['test_precision']))

data2 = data.pivot(index='kernel', columns='cost' , values='test_f1_score')

heatmap = sb.heatmap(data2)
# plt.title('Heatmap of Algorithm Performance')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend().set_title('')

plt.savefig('charts/{algorithm}_heatmap.png'.format(algorithm=args.algorithm))