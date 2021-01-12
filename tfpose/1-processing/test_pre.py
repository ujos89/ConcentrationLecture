#preprocessing top, bottom
import argparse
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


#parser control
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, default='', help='raw (pickle)data path and name without ".pkl"')
parser.add_argument('--name', type=str, default='', help='merge pickle data by name')
parser.add_argument('--index', type=str, default=0, help='index to indicate dataframe')
args = parser.parse_args()

def mergebyname(name):
    #filename to merge
    files = os.listdir('../0-data/data_pickle/')
    file2merge = []
    for _ in files:
        if _.startswith(name):
            file2merge.append(_)
    
    #merged dataset by name
    df_merged = pd.DataFrame()
    for _ in file2merge:
        df_tmp = pd.read_pickle('../0-data/data_pickle/'+_)
        df_merged = pd.concat([df_merged, df_tmp])

    return df_merged

def rearrange(df_raw):
    #body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rhi", "Rkn", "Ran", "Lhi", "Lkn", "Lan", "Rey", "Ley", "Rea", "Lea"]
    top = ["Nos_X","Nos_Y","Ley_X","Ley_Y","Rey_X","Rey_Y","Lea_X","Lea_Y","Rea_X","Rea_Y"]
    mid = ["Nec_X","Nec_Y","Lsh_X","Lsh_Y","Rsh_X","Rsh_Y","Lel_X","Lel_Y","Rel_X","Rel_Y"]

    return df_raw[top], df_raw[mid]

#get variance by divided data
def get_var(df, cut_num):
    #(X,Y)*5 for plot
    data = [np.array([])for _ in range(10)]
    #x,y variance for training
    x_var = np.array([])
    y_var = np.array([])
    i = 0
    scaler = preprocessing.MinMaxScaler()
    
    while i <= len(df):
        #cut data to get variacne
        if i + cut_num <= len(df):
            df_tmp = df[i:i+cut_num]
        #remove remained data after cut
        else:
            break

        X_data = np.array([])
        Y_data = np.array([])

        #calculate for each X,Y
        for idx in range(10):
            #process missing value
            cutmv = np.array(df_tmp.iloc[:, idx].loc[(df!=0).any(axis=1)]).flatten()
            #get data after process missing value (X,Y value for each parts)
            data[idx] = np.append(data[idx], cutmv)

            #Min-Max Scaler (when data size is bigger than 1)
            if len(cutmv) > 1:
                scaled = scaler.fit_transform(cutmv.reshape(-1,1))
            else:                
                scaled = np.array([])
            
            #add to X, Y
            if idx%2:
                Y_data = np.append(Y_data, scaled)
            else:
                X_data = np.append(X_data, scaled)

        #get variance
        x_var = np.append(x_var, np.var(X_data))
        y_var = np.append(y_var, np.var(Y_data))

        i += cut_num
        
    return data, x_var, y_var

##main

#case1: pickle file -> preprocessing
if args.file:
    df_raw = pd.read_pickle('../0-data/data_pickle/'+args.file+'.pkl')
#case2: pickle files(merge by name) -> preprocessing
elif args.name:
    df_raw = mergebyname(args.name)

df_top, df_mid = rearrange(df_raw)

#get var from top & mid
data_top, top_x_var, top_y_var = get_var(df_top, 1000)
data_mid, mid_x_var, mid_y_var = get_var(df_top, 1000)

