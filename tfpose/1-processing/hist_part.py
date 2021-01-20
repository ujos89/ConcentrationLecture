import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# parser
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--scaler', type=int, default=0, help='0-nonscaler 1-scaler')
parser.add_argument('--num', type=int, default=100, help='per N frames')
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_pickle/'+args.c)       
df_nc = pd.read_pickle('../0-data/data_pickle/'+args.nc)      

# neckx, necky
fig, axes = plt.subplots(nrows=2, ncols=2)      
nbin = 50

part = 'Rey'

df_c = df_c[[part + '_X', part + '_Y']]
df_nc = df_nc[[part + '_X', part + '_Y']]

def getStd(data, num):
    stdX = np.array([])
    stdY = np.array([])
    lend = len(data)
    idx = 0
    while (idx < lend):
        tmp = data[idx:idx+num]
        lent = len(tmp)
        if (lent > 1):
            stdX = np.append(stdX, (np.std(tmp[part + '_X'])))
            stdY = np.append(stdY, (np.std(tmp[part + '_Y'])))
            idx += num
        else:
            break

    return pd.DataFrame(stdX, columns=['stdX']), pd.DataFrame(stdY, columns=['stdY'])
    
# define drawing std 1-D histogram function
def drawStdHist(dataX, dataY, row, rangemin, rangemax):
    for i in range(2):
        if i == 0:
            data = dataX
            label = part + '_X'
        else:
            data = dataY
            label = part + '_Y'

        axes[row, i].hist(data, range=(rangemin, rangemax), bins=nbin)
        axes[row, i].set_xlabel(label + '_' + str(row), fontsize=10)
        axes[row, i].set_ylabel('Num', fontsize=10)

c_stdX, c_stdY = getStd(df_c, args.num)
nc_stdX, nc_stdY = getStd(df_nc, args.num)

df_c = pd.concat([c_stdX, c_stdY], axis=1)
df_c['label'] = 1
df_nc = pd.concat([nc_stdX, nc_stdY], axis=1)
df_nc['label'] = 0

if args.scaler:
    df = pd.concat([df_c, df_nc])
    
    for i in df.columns[:-1]:
        df[i] = (df[i] - df[i].mean()) / df[i].std()
    
    df_c = df[df['label'] == 1]
    df_nc = df[df['label'] == 0]

print(df_c)
print(df_nc)

print(df_c.describe())
print(df_nc.describe())


drawStdHist(df_c['stdX'], df_c['stdY'], 1, -1, 1)        # concentrate
drawStdHist(df_nc['stdX'], df_nc['stdY'], 0, -1, 1)      # not concentrate


plt.show()
