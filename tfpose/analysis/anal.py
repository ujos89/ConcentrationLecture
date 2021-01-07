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

df1_0 = dfRaw.loc[lambda df: df['label(1-fold)']==0 , lambda df : ['predcition(1-fold)', 'label(1-fold)']]
df1_1 = dfRaw.loc[lambda df: df['label(1-fold)']==1 , lambda df : ['predcition(1-fold)', 'label(1-fold)']]

print(df1_0)
print(df1_1)

