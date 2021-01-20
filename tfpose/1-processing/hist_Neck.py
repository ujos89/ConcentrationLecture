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
nbin = 20

df_c = df_c[['Nec_X', 'Nec_Y']]
df_nc = df_nc[['Nec_X', 'Nec_Y']]

def getStd(data, num):
    stdX = np.array([])
    stdY = np.array([])
    lend = len(data)
    idx = 0
    while (idx < lend):
        tmp = data[idx:idx+num]
        lent = len(tmp)
        if (lent > 1):
            stdX = np.append(stdX, (np.std(tmp['Nec_X'])))
            stdY = np.append(stdY, (np.std(tmp['Nec_Y'])))
            idx += num
        else:
            break

    return stdX, stdY
    
# define drawing std 1-D histogram function
def drawStdHist(dataX, dataY, row, rangemin, rangemax):
    for i in range(2):
        if i == 0:
            data = dataX
            label = 'Nec_X'
        else:
            data = dataY
            label = 'Nec_Y'

        axes[row, i].hist(data, range=(rangemin, rangemax), bins=nbin)
        axes[row, i].set_xlabel(label + '_' + str(row), fontsize=15)
        axes[row, i].set_ylabel('Num', fontsize=15)

c_stdX, c_stdY = getStd(df_c, args.num)
nc_stdX, nc_stdY = getStd(df_nc, args.num)

if args.scaler:
    standardScaler = StandardScaler()
    c_stdX = (standardScaler.fit_transform(c_stdX.reshape(-1, 1)))
    c_stdY = (standardScaler.fit_transform(c_stdY.reshape(-1, 1)))
    nc_stdX = (standardScaler.fit_transform(nc_stdX.reshape(-1, 1)))
    nc_stdY = (standardScaler.fit_transform(nc_stdY.reshape(-1, 1)))
    
drawStdHist(c_stdX, c_stdY, 1, 0, 0.5)        # concentrate
drawStdHist(nc_stdX, nc_stdY, 0, 0, 0.5)      # not concentrate

plt.show()
