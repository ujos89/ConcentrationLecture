import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


dataPath = '../0-data/data_prediction/4R16R1S_es_100/90_for_kf.pkl'

dfRaw = pd.read_pickle(dataPath)


print(dfRaw)


tFrame =  50.
tFps = 20.
tWin = tFrame * (1/tFps)
tMax = tWin * len(dfRaw)


n4error = tFrame

error = 12 / np.sqrt(n4error)



# print(error)

dfMeasure = dfRaw['prediction']

# print(dfMeasure)


dfMeasureCut = pd.DataFrame() 

nCut = 1

for i in range(0, len(dfRaw), nCut) :
    dfMeasureOne =  pd.DataFrame([dfMeasure.loc[i]])
    dfMeasureCut = pd.concat([dfMeasureCut, dfMeasureOne])

# print(dfMeasureCut)





tWinCut = nCut * tWin

tNp = np.arange(0, tMax, step=tWinCut)

u = np.array([[dfMeasureCut.iloc[1]], [dfMeasureCut.iloc[1]]])

# print(dfMeasureCut.iloc[1])

# print(u)

# print(tNp)

plt.scatter(tNp, dfRaw['prediction'], c=dfRaw['label'])

plt.show()
