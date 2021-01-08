import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import printColor as pc 
from scipy.optimize import curve_fit

color = pc.bcolors()
#Dictionizing dataframes - Raw data
def funcChop(dfDictRaw, dfRaw):
    #dfDictRaw = dict(('dfRaw' + str(x), pd.DataFrame()) for x in range(0, 51 , 3))
    #Chopping data
    for i in range(0, 51 , 3):    
        name = 'dfRaw' + str(i)
        dfTemp = dfRaw.iloc[:,i:i+3]
        dfTemp = dfTemp[dfTemp != 0.0].dropna()
        dfDictRaw[name] = dfTemp
    return dfDictRaw

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

#inserting normalization data
def funcNorm(dfDictNorm, dfDictRaw, dfRaw):
    #list for empty dataframe
    dfEmpty = []
    for i in range(0, 51 , 3):    
        nameRaw = 'dfRaw' + str(i)
        nameNorm = 'dfNorm' + str(i)
        #print(len(dfDictRaw[nameRaw]))
        if len(dfDictRaw[nameRaw]) > 100 :
            dfDictNorm[nameNorm] = funcMs(dfDictRaw[nameRaw])
        else :
            print(i)
            dfEmpty.append(i)
    #sorting empty index in RawData
    indexEmpty = dfRaw.columns[dfEmpty].values
    if len(indexEmpty) !=0 : 
        print(color.RED + str(indexEmpty) + "will not use"+ color.ENDC)
    else : 
        print(color.RED + "all data is nomalized"+ color.ENDC)

    return dfDictNorm

def fitGaussian(x,a,mean,sigma):
       return (a*np.exp(-((x-mean)**2/(2*sigma))))
       

