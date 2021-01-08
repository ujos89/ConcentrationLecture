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
#dfRaw = dfRaw.iloc[:300]
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
fig1, axes1 = plt.subplots(nrows=2, ncols=6)

nbins = 25

#Top
#nose
dfNormNoseX = funcSs(dfDictRaw["dfRaw0"]).iloc[:,0]
dfNormNoseY = funcSs(dfDictRaw["dfRaw0"]).iloc[:,1]
axes1[0, 0].set_title(indexTotal[0])
axes1[0, 0].hist2d( dfNormNoseX,  dfNormNoseY, bins=nbins)

#Left Eye
dfNormLeEyeX = funcSs(dfDictRaw["dfRaw45"]).iloc[:,0]
dfNormLeEyeY = funcSs(dfDictRaw["dfRaw45"]).iloc[:,1]
axes1[0, 1].set_title(indexTotal[45])
axes1[0, 1].hist2d( dfNormLeEyeX,  dfNormLeEyeY, bins=nbins)

#Right Eye
dfNormRiEyeX = funcSs(dfDictRaw["dfRaw42"]).iloc[:,0]
dfNormRiEyeY = funcSs(dfDictRaw["dfRaw42"]).iloc[:,1]
axes1[0, 2].set_title(indexTotal[42])
axes1[0, 2].hist2d( dfNormRiEyeX,  dfNormRiEyeY, bins=nbins)

#Right ear
dfNormRiEerX = funcSs(dfDictRaw["dfRaw48"]).iloc[:,0]
dfNormRiEerY = funcSs(dfDictRaw["dfRaw48"]).iloc[:,1]
axes1[0, 3].set_title(indexTotal[48])
axes1[0, 3].hist2d( dfNormRiEerX,  dfNormRiEerY, bins=nbins)

#Concating Top part 2D
dfNormTopX = pd.concat([dfNormNoseX,dfNormLeEyeX,dfNormRiEyeX,dfNormRiEerX ] ,axis=0)
dfNormTopY = pd.concat([dfNormNoseY,dfNormLeEyeY,dfNormRiEyeY,dfNormRiEerY ] ,axis=0)

axes1[0, 4].set_title("Top")
axes1[0, 4].hist2d( dfNormTopX,  dfNormTopY, bins=nbins)

#Concating Top part 1D
dfNormTop = pd.concat([dfNormTopX, dfNormTopY] ,axis=0)

axes1[0, 5].set_title("Top")
axes1[0, 5].hist( dfNormTop, bins=nbins)

#middle
#Neck
dfNormNeckX = funcSs(dfDictRaw["dfRaw3"]).iloc[:,0]
dfNormNeckY = funcSs(dfDictRaw["dfRaw3"]).iloc[:,1]
axes1[1, 0].set_title(indexTotal[3])
axes1[1, 0].hist2d( dfNormNeckX,  dfNormNeckY, bins=nbins)

#Right shoulder
dfNormRiShX = funcSs(dfDictRaw["dfRaw6"]).iloc[:,0]
dfNormRishY = funcSs(dfDictRaw["dfRaw6"]).iloc[:,1]
axes1[1, 1].set_title(indexTotal[6])
axes1[1, 1].hist2d( dfNormRiShX,  dfNormRishY, bins=nbins)

#Left shoulder
dfNormLeShX = funcSs(dfDictRaw["dfRaw15"]).iloc[:,0]
dfNormLeshY = funcSs(dfDictRaw["dfRaw15"]).iloc[:,1]
axes1[1, 2].set_title(indexTotal[15])
axes1[1, 2].hist2d( dfNormLeShX,  dfNormLeshY, bins=nbins)

#Concating Mid part 2D
dfNormMidX = pd.concat([dfNormNeckX,dfNormRiShX,dfNormLeShX ] ,axis=0)
dfNormMidY = pd.concat([dfNormNeckY,dfNormRishY,dfNormLeshY ] ,axis=0)
axes1[1, 4].set_title("Mid")
axes1[1, 4].hist2d( dfNormMidX,  dfNormMidY, bins=nbins)

#Concating Mid part 1D
dfNormMid = pd.concat([dfNormMidX, dfNormMidY] ,axis=0)
axes1[1, 5].set_title("Mid")
axes1[1, 5].hist( dfNormMid, bins=nbins)


#fig2, axes2 = plt.subplots(nrows=2, ncols=6)

# for ax, gamma in zip(axes.flat[1:], gammas):
#     ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
#     ax.hist2d(data[:, 0], data[:, 1],
#               bins=100, norm=mcolors.PowerNorm(gamma))

fig1.tight_layout()

plt.show()

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kjk_C03_1.pkl
