
import argparse
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, help='algorithm', default=None)
args = parser.parse_args()
print(args)

data = pd.read_csv('output_cv/{args.algorithm}_cross_validation.csv')
heatmap = sb.heatmap(data)
# plt.title('Heatmap of Algorithm Performance')
# plt.xlabel('')
# plt.ylabel('')
# plt.legend().set_title('')

plt.savefig('charts/{args.algorithm}_heatmap.png')