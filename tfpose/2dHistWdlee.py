import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas_profiling

#parsing data
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

#import to dataFrame
dfRaw = pd.read_pickle(args.file)      # x, y, score
# dfRaw = dfRaw.iloc[:300]
#print(dfRaw.iloc[:,0])

#Dictionizing dataframes - Raw data
dfDictRaw = dict(('dfRaw' + str(x), pd.DataFrame()) for x in range(0, 51 , 3))

#Chopping data
for i in range(0, 51 , 3):    
    name = 'dfRaw' + str(i)
    dfTemp = dfRaw.iloc[:,i:i+3]
    dfTemp = dfTemp[dfTemp != 0.0].dropna()
    dfDictRaw[name] = dfTemp
#print(dfDictRaw)

#Normalization functions

#Min-Max Scaler
def funcMs(dfRaw):
    xRaw = dfRaw.values.astype(float)
    mimMaxScaler = preprocessing.MinMaxScaler()
    xScaled = mimMaxScaler.fit_transform(xRaw)
    dfNorm = pd.DataFrame(xScaled, columns=dfRaw.columns)
    return dfRaw

#Standard Scaler
def funcSs(dfRaw):
    StandardScaler().fit(dfRaw)
    npNorm = StandardScaler().fit_transform(dfRaw)
    dfNorm = pd.DataFrame(npNorm, columns=dfRaw.columns)
    return dfNorm

#Dictionizing dataframes - Nomalizing data
dfDictNorm = dict(('dfNorm' + str(x), pd.DataFrame()) for x in range(0, 51 , 3))

#list for empty dataframe
dfEmpty = []

#inserting normalization data
for i in range(0, 51 , 3):    
    nameRaw = 'dfRaw' + str(i)
    nameNorm = 'dfNorm' + str(i)
    #print(len(dfDictRaw[nameRaw]))
    if len(dfDictRaw[nameRaw]) > 100 :
        dfDictNorm[nameNorm] = funcSs(dfDictRaw[nameRaw])
    else :
        print(i)
        dfEmpty.append(i)

#total index
indexTotal = dfRaw.columns.values
print(indexTotal)

#sorting empty index in RawData
indexEmpty = dfRaw.columns[dfEmpty].values



dfNorm = funcSs(dfDictRaw["dfRaw0"])

#extracting X, Y data from normalized data
dfNormX = dfNorm.iloc[:,0]
dfNormY = dfNorm.iloc[:,1]
#print(dfNorm)

rangeMin = -1
rangeMax = 1

#profile
# pr = dfRaw.profile_report()
# pr.to_file('./testResult.html')

#2d Histogram
fig, axes = plt.subplots(nrows=3, ncols=3)

nbins = 50

axes[0, 0].set_title(indexTotal[0])
axes[0, 0].hist2d( funcSs(dfDictRaw["dfRaw0"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw0"]).iloc[:,1], bins=nbins)

axes[1, 0].set_title(indexTotal[3])
axes[1, 0].hist2d( funcSs(dfDictRaw["dfRaw3"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw3"]).iloc[:,1], bins=nbins)

axes[2, 0].set_title(indexTotal[6])
axes[2, 0].hist2d( funcSs(dfDictRaw["dfRaw6"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw6"]).iloc[:,1], bins=nbins)

axes[0, 1].set_title(indexTotal[15])
axes[0, 1].hist2d( funcSs(dfDictRaw["dfRaw15"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw15"]).iloc[:,1], bins=nbins)

axes[0, 2].set_title(indexTotal[45])
axes[0, 2].hist2d( funcSs(dfDictRaw["dfRaw45"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw45"]).iloc[:,1], bins=nbins)

axes[1, 1].set_title(indexTotal[48])
axes[1, 1].hist2d( funcSs(dfDictRaw["dfRaw48"]).iloc[:,0],  funcSs(dfDictRaw["dfRaw48"]).iloc[:,1], bins=nbins)


# for ax, gamma in zip(axes.flat[1:], gammas):
#     ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
#     ax.hist2d(data[:, 0], data[:, 1],
#               bins=100, norm=mcolors.PowerNorm(gamma))

fig.tight_layout()

plt.show()

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kjk_C03_1.pkl
