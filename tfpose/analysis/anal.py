import pandas as pd
import argparse
import matplotlib.pyplot as plt
 
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--path', type=str, required=True, help='pickle file to read')
args = parser.parse_args()

dfRaw = pd.read_pickle(args.path)
print(dfRaw)

nBins= 25

plt.subplot(3,2,1)
plt.hist(dfRaw.loc[lambda df: df['label(1-fold)']==0, 'predcition(1-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.hist(dfRaw.loc[lambda df: df['label(1-fold)']==1, 'predcition(1-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,2)
plt.hist(dfRaw.loc[lambda df: df['label(2-fold)']==0, 'predcition(2-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.hist(dfRaw.loc[lambda df: df['label(2-fold)']==1, 'predcition(2-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,3)
plt.hist(dfRaw.loc[lambda df: df['label(3-fold)']==0, 'predcition(3-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.hist(dfRaw.loc[lambda df: df['label(3-fold)']==1, 'predcition(3-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,4)
plt.hist(dfRaw.loc[lambda df: df['label(4-fold)']==0, 'predcition(4-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.hist(dfRaw.loc[lambda df: df['label(4-fold)']==1, 'predcition(4-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,5)
plt.hist(dfRaw.loc[lambda df: df['label(5-fold)']==0, 'predcition(5-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.hist(dfRaw.loc[lambda df: df['label(5-fold)']==1, 'predcition(5-fold)'], bins=nBins,density=True, alpha=0.5, histtype='step' )
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.show()

