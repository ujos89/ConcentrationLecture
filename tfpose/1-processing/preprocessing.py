import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

#parser control
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--name', type=str, default='', help='preprocessing all file by name in folder')
parser.add_argument('--cut', type=int, default=100, help='standard for cut to get std')
args = parser.parse_args()

def rearrange(df_raw):
    #body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rhi", "Rkn", "Ran", "Lhi", "Lkn", "Lan", "Rey", "Ley", "Rea", "Lea"]
    top = ["Nos_X","Nos_Y","Ley_X","Ley_Y","Rey_X","Rey_Y","Lea_X","Lea_Y","Rea_X","Rea_Y"]
    mid = ["Nec_X","Nec_Y","Lsh_X","Lsh_Y","Rsh_X","Rsh_Y","Lel_X","Lel_Y","Rel_X","Rel_Y"]

    return df_raw[top], df_raw[mid]

#get variance by divided data
def get_std(df, cut):
    #(X,Y)*5 for plot
    data = [np.array([])for _ in range(10)]
    #x,y variance for training
    x_std = np.array([])
    y_std = np.array([])
    i = 0
    scaler = preprocessing.MinMaxScaler()
    
    while i <= len(df):
        #cut data to get variacne
        if i + cut <= len(df):
            df_tmp = df[i:i+cut]
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

        #get standard deviation
        x_std = np.append(x_std, np.std(X_data))
        y_std = np.append(y_std, np.std(Y_data))

        i += cut

    #output data type: numpy array    
    return data, x_std, y_std

def file_preprocessing(file_, cut):
    #read pickle
    df_raw = pd.read_pickle('../0-data/data_pickle/'+file_)
    df_top, df_mid = rearrange(df_raw)

    #get standard deviation from top & mid
    data_top, top_x_std, top_y_std = get_std(df_top, cut)
    data_mid, mid_x_std, mid_y_std = get_std(df_mid, cut)

    #data merge for dnn
    input_data = np.concatenate((top_x_std.reshape(-1,1), top_y_std.reshape(-1,1), mid_x_std.reshape(-1,1), mid_y_std.reshape(-1,1)), axis=1)
    df_prepared = pd.DataFrame(input_data, columns=['Top_X','Top_Y','Mid_X','Mid_Y'])
    #add label
    df_prepared['label'] = int(file_[-5])

    #output datatype: pandas datafame
    return df_prepared


##main

#case1: preprocessing for file
if args.file:
    df_prepared = file_preprocessing(args.file, args.cut)

    #save to pickle
    df_prepared.to_pickle('../0-data/data_prepared/'+args.file[:-6]+'.pkl')

elif args.name:
    #file list to merge
    file_list = []
    files = os.listdir('../0-data/data_pickle')
    for _ in files:
        if _.startswith(args.name):
            file_list.append(_)
    
    #preprocessing and merge
    df_merged = pd.DataFrame()
    for _ in file_list:
        df_prepared = file_preprocessing(_, args.cut)
        df_merged = pd.concat([df_merged, df_prepared])

    #shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    #save to pickle
    df_merged.to_pickle('../0-data/data_prepared/merged/'+args.name+'_merged.pkl')