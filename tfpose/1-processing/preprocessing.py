import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing

#parser control
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='raw (pickle)data path and name without ".pkl"')
args = parser.parse_args()

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

#read pickle
df_raw = pd.read_pickle('../0-data/data_pickle/'+args.file+'.pkl')
df_top, df_mid = rearrange(df_raw)

#get var from top & mid
data_top, top_x_var, top_y_var = get_var(df_top, 1000)
data_mid, mid_x_var, mid_y_var = get_var(df_mid, 1000)

#data merge for dnn
input_data = np.concatenate((top_x_var.reshape(-1,1), top_y_var.reshape(-1,1), mid_x_var.reshape(-1,1), mid_y_var.reshape(-1,1)), axis=1)
df_prepared = pd.DataFrame(input_data, columns=['Top_X','Top_Y','Mid_X','Mid_Y'])
#add label
df_prepared['label'] = int(args.file[-1])

#save to pickle
df_prepared.to_pickle('../0-data/data_prepared/'+args.file[:-2]+'.pkl')

print(df_prepared)