import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import printColor as pc 
import modules as md
import scipy

#color import
color = pc.bcolors()

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

#import to dataFrame
dfRaw = pd.read_pickle(args.file) 

#if you want to cut
dfRaw = dfRaw.iloc[:10000]
print(color.GREEN + str(args) + "is ready"+ color.ENDC)


#Dictionizing dataframes - Raw data
dfDictRaw = dict(('dfRaw' + str(x), pd.DataFrame()) for x in range(0, 51 , 3))

#Chopping data
dfDictRaw = md.funcChop(dfDictRaw, dfRaw)


#Dictionizing dataframes - Normalizing data
dfDictNorm = dict(('dfNorm' + str(x), pd.DataFrame()) for x in range(0, 51 , 3))

#normalizing & checking empty data 
dfDictNorm = md.funcNorm(dfDictNorm, dfDictRaw, dfRaw)

#print(dfDictNorm)
#2d Histogram number
histNum = len(dfDictNorm)
print(histNum)
print(dfDictNorm['dfNorm39'])

fig1, axes1 = plt.subplots(nrows=3, ncols=6)


nbins = 25

#TopHisto
dfNormTopX = pd.DataFrame()
dfNormTopY = pd.DataFrame()
for i in range(0,4):
  dfName = "dfNorm"+str(i*3) 
  if len(dfDictNorm[dfName]) !=0 :
    dfNormX = dfDictNorm[dfName].iloc[:,0]
    dfNormY = dfDictNorm[dfName].iloc[:,1]
    dfNormTopX = pd.concat([dfNormTopX, dfNormX] ,axis=0)
    dfNormTopY = pd.concat([dfNormTopY, dfNormY] ,axis=0)
    name = dfDictNorm[dfName].columns
    axes1[0, i].hist2d( dfNormX,  dfNormY, bins=nbins, range = [[0, 1], [0, 1]])
    axes1[0, i].set_xlabel(name[0])
    axes1[0, i].set_ylabel(name[1])

axes1[0, 4].hist2d(dfNormTopX.iloc[:,0],  dfNormTopY.iloc[:,0], bins=nbins, range = [[0, 1], [0, 1]])
axes1[0, 4].set_xlabel('TopX')
axes1[0, 4].set_ylabel('TopY')

dfNormTopAll = pd.concat([dfNormTopX,dfNormTopY], axis=0)
# print(dfNormTopAll.iloc[:,0])
axes1[0, 5].hist(dfNormTopAll.iloc[:,0], bins=50, range=(0,1))
axes1[0, 5].set_xlabel('Top')

#MidHisto
dfNormMidX = pd.DataFrame()
dfNormMidY = pd.DataFrame()
for i in range(0,4):
  dfName = "dfNorm"+str(i*3 + 12)
  if len(dfDictNorm[dfName]) !=0 :
    dfNormX = dfDictNorm[dfName].iloc[:,0]
    dfNormY = dfDictNorm[dfName].iloc[:,1]
    dfNormMidX = pd.concat([dfNormMidX, dfNormX] ,axis=0)
    dfNormMidY = pd.concat([dfNormMidY, dfNormY] ,axis=0)
    name = dfDictNorm[dfName].columns
    axes1[1, i].hist2d( dfNormX,  dfNormY, bins=nbins, range = [[0, 1], [0, 1]])
    axes1[1, i].set_xlabel(name[0])
    axes1[1, i].set_ylabel(name[1])

axes1[1, 4].hist2d(dfNormMidX.iloc[:,0],  dfNormMidY.iloc[:,0], bins=nbins, range = [[0, 1], [0, 1]])
axes1[1, 4].set_xlabel('MidX')
axes1[1, 4].set_ylabel('MidY')

dfNormMidAll = pd.concat([dfNormMidX,dfNormMidY], axis=0)
# print(dfNormMidAll)
axes1[1, 5].hist(dfNormMidAll.iloc[:,0], bins=50, range=(0,1))
axes1[1, 5].set_xlabel('Mid')

#BotHisto
dfNormBotX = pd.DataFrame()
dfNormBotY = pd.DataFrame()
for i in range(0,4):
  dfName = "dfNorm"+str(i*3 + 24) 
  if len(dfDictNorm[dfName]) !=0 :
    dfNormX = dfDictNorm[dfName].iloc[:,0]
    dfNormY = dfDictNorm[dfName].iloc[:,1]
    dfNormBotX = pd.concat([dfNormBotX, dfNormX] ,axis=0)
    dfNormBotY = pd.concat([dfNormBotY, dfNormY] ,axis=0)
    name = dfDictNorm[dfName].columns
    axes1[2, i].hist2d( dfNormX,  dfNormY, bins=nbins, range = [[0, 1], [0, 1]])
    axes1[2, i].set_xlabel(name[0])
    axes1[2, i].set_ylabel(name[1])

axes1[2, 4].hist2d(dfNormBotX.iloc[:,0],  dfNormBotY.iloc[:,0], bins=nbins, range = [[0, 1], [0, 1]])
axes1[2, 4].set_xlabel('BotX')
axes1[2, 4].set_ylabel('BotY')

dfNormBotAll = pd.concat([dfNormBotX,dfNormBotY], axis=0)
# print(dfNormBotAll)
axes1[2, 5].hist(dfNormBotAll.iloc[:,0], bins=50, range=(0,1))
axes1[2, 5].set_xlabel('Bot')



plt.show()
