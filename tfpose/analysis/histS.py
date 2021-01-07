import pandas as pd
import argparse
import matplotlib.pyplot as plt
 
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--path', type=str, required=True, help='pickle file to read')
args = parser.parse_args()

dfRaw = pd.read_pickle(args.path)

nBins= 100
rangeMin = 0
rangeMax = 1

dfAll = pd.DataFrame()

for i in range(1,6):
    namePre = 'predcition(' + str(i) + '-fold)'
    nameLabel = 'label(' + str(i) + '-fold)'
    dfTemp = dfRaw[[namePre, nameLabel]]
    dfTemp.columns=['prediction','label']
    dfAll = pd.concat([dfAll, dfTemp])

print(dfRaw)
print(dfAll)

plt.subplot(3,2,1)
plt.hist(dfRaw.loc[lambda df: df['label(1-fold)']==0, 'predcition(1-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfRaw.loc[lambda df: df['label(1-fold)']==0, 'predcition(1-fold)'])))
plt.hist(dfRaw.loc[lambda df: df['label(1-fold)']==1, 'predcition(1-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfRaw.loc[lambda df: df['label(1-fold)']==1, 'predcition(1-fold)'])))
plt.hist(dfRaw['predcition(1-fold)'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfRaw['predcition(1-fold)'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,2)
plt.hist(dfRaw.loc[lambda df: df['label(2-fold)']==0, 'predcition(2-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfRaw.loc[lambda df: df['label(2-fold)']==0, 'predcition(2-fold)'])))
plt.hist(dfRaw.loc[lambda df: df['label(2-fold)']==1, 'predcition(2-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfRaw.loc[lambda df: df['label(2-fold)']==1, 'predcition(2-fold)'])))
plt.hist(dfRaw['predcition(2-fold)'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfRaw['predcition(2-fold)'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])  

plt.subplot(3,2,3)
plt.hist(dfRaw.loc[lambda df: df['label(3-fold)']==0, 'predcition(3-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfRaw.loc[lambda df: df['label(3-fold)']==0, 'predcition(3-fold)'])))
plt.hist(dfRaw.loc[lambda df: df['label(3-fold)']==1, 'predcition(3-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfRaw.loc[lambda df: df['label(3-fold)']==1, 'predcition(3-fold)'])))
plt.hist(dfRaw['predcition(3-fold)'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfRaw['predcition(3-fold)'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])   

plt.subplot(3,2,4)
plt.hist(dfRaw.loc[lambda df: df['label(4-fold)']==0, 'predcition(4-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfRaw.loc[lambda df: df['label(4-fold)']==0, 'predcition(4-fold)'])))
plt.hist(dfRaw.loc[lambda df: df['label(4-fold)']==1, 'predcition(4-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfRaw.loc[lambda df: df['label(4-fold)']==1, 'predcition(4-fold)'])))
plt.hist(dfRaw['predcition(4-fold)'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfRaw['predcition(4-fold)'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])    

plt.subplot(3,2,5)
plt.hist(dfRaw.loc[lambda df: df['label(5-fold)']==0, 'predcition(5-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfRaw.loc[lambda df: df['label(5-fold)']==0, 'predcition(5-fold)'])))
plt.hist(dfRaw.loc[lambda df: df['label(5-fold)']==1, 'predcition(5-fold)'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfRaw.loc[lambda df: df['label(5-fold)']==1, 'predcition(5-fold)'])))
plt.hist(dfRaw['predcition(5-fold)'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfRaw['predcition(5-fold)'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])   

plt.subplot(3,2,6)
plt.hist(dfAll.loc[lambda df: df['label']==0, 'prediction'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "black", label='label 0, ' + str(len(dfAll.loc[lambda df: df['label']==0, 'prediction'])))
plt.hist(dfAll.loc[lambda df: df['label']==1, 'prediction'], bins=nBins, range=(rangeMin,rangeMax), alpha=1, histtype='step',color = "red",   label='label 1, ' + str(len(dfAll.loc[lambda df: df['label']==1, 'prediction'])))
plt.hist(dfAll['prediction'], bins=nBins, range=(rangeMin,rangeMax),alpha=0.5, histtype='stepfilled', color = "skyblue",label='label all, '+ str(len(dfAll['prediction'])) )
plt.legend(loc='upper left')
plt.xlabel('Prediction Value')
plt.ylabel('Numbers')
plt.xlim([0, 1])   



plt.show()

