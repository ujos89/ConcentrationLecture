import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# parser
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_prepared/'+args.c)        # ( , 5)
df_nc = pd.read_pickle('../0-data/data_prepared/'+args.nc)      # ( , 5)

# topX, topY, midX, midY, top, mid, total
fig, axes = plt.subplots(nrows=2, ncols=7)      

# define concatenation function
def concatData(data):
    dataT = pd.concat([data.iloc[:, 0], data.iloc[:, 1]])
    dataM = pd.concat([data.iloc[:, 2], data.iloc[:, 3]])
    dataTotal = pd.concat([dataT, dataM])
    
    return dataT, dataM, dataTotal

# define drawing std 1-D histogram function
def drawStdHist(data, row):
    dataT, dataM, dataTotal = concatData(data)    
    for i in range(4):
        axes[row, i].hist(data.iloc[i])
        axes[row, i].set_xlabel(data.columns[i] + '_' + str(row), fontsize=10)
        axes[row, i].set_ylabel('Num', fontsize=10)
    
    axes[row, 4].hist(dataT)
    axes[row, 4].set_xlabel('Top_' + str(row), fontsize=10)
    axes[row, 4].set_ylabel('Num', fontsize=10)

    axes[row, 5].hist(dataM)
    axes[row, 5].set_xlabel('Mid_' + str(row), fontsize=10)
    axes[row, 5].set_ylabel('Num', fontsize=10)
    
    axes[row, 6].hist(dataTotal)
    axes[row, 6].set_xlabel('Total_' + str(row), fontsize=10)
    axes[row, 6].set_ylabel('Num', fontsize=10)


# draw
drawStdHist(df_c, 1)
drawStdHist(df_nc, 0)

plt.show()

