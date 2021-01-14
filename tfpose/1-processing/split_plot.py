import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
args = parser.parse_args()

df = pd.read_pickle('../0-data/data_prepared/'+args.file)        # ( , 5)

# split
df_c = df[df['label']==1]
df_nc = df[df['label']==0]

fig, axes = plt.subplots(nrows=2, ncols=4)      
nbin = 50

def draw1dHist(data, row):
    for i in range(4):
        axes[row, i].hist(data[data.columns[i]], bins=nbin, range=(0.01, 0.3))
        axes[row, i].set_xlabel(data.columns[i] + '_' + str(row))
        axes[row, i].set_ylabel('Num')

draw1dHist(df_c, 1)
draw1dHist(df_nc, 0)

print(df_c.describe())
print(df_nc.describe())

plt.show()

