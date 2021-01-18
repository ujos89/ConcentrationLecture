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
    #x,y variance for training
    x_std = np.array([])
    y_std = np.array([])
    #scaler = preprocessing.MinMaxScaler()
    i, status = 0, 0
    len_df = len(df)

    while i <= len_df:
        #cut data to get variacne
        if i + cut <= len_df:
            df_tmp = df[i:i+cut]
        #remove remained data after cut
        else:
            break

        X_data = np.array([])
        Y_data = np.array([])

        #calculate for each X,Y
        for idx in range(10):
            #process missing value
            cutmv = np.array(df_tmp.iloc[:, idx].loc[(df_tmp!=0).any(axis=1)]).flatten()

            #make mean of value to zero
            if len(cutmv) > 0:
                scaled = cutmv - np.mean(cutmv)
                #Min-Max Scaler (when data size is bigger than 1)
                #scaled = scaler.fit_transform(cutmv.reshape(-1,1))
            else:
                #If all value in cut are 0, replace with Nan
                scaled = np.array([])
            
            #add to X, Y
            if idx%2:
                Y_data = np.append(Y_data, scaled)
            else:
                X_data = np.append(X_data, scaled)

        #get standard deviation
        x_std = np.append(x_std, np.std(X_data))
        y_std = np.append(y_std, np.std(Y_data))

        #print status progress
        if(status < i/len_df*10):
            status += 1
            print("Progress: ", (i/len_df*100)//1, "%")

        i += cut

    #output data type: numpy array    
    return x_std, y_std

def file_preprocessing(file_, cut):
    #read pickle
    df_raw = pd.read_pickle('../0-data/data_pickle/'+file_)
    df_top, df_mid = rearrange(df_raw)

    #get standard deviation from top & mid
    top_x_std, top_y_std = get_std(df_top, cut)
    mid_x_std, mid_y_std = get_std(df_mid, cut)

    #data merge for dnn
    input_data = np.concatenate((top_x_std.reshape(-1,1), top_y_std.reshape(-1,1), mid_x_std.reshape(-1,1), mid_y_std.reshape(-1,1)), axis=1)
    df_prepared = pd.DataFrame(input_data, columns=['Top_X','Top_Y','Mid_X','Mid_Y'])
    #add label
    df_prepared['label'] = int(file_[-5])

    #output datatype: pandas datafame
    return df_prepared

def rearrange_dis(df_raw):
    use = ["Nos_X","Nos_Y","Ley_X","Ley_Y","Rey_X","Rey_Y","Lea_X","Lea_Y","Rea_X","Rea_Y","Nec_X","Nec_Y","Lsh_X","Lsh_Y","Rsh_X","Rsh_Y","Lel_X","Lel_Y","Rel_X","Rel_Y"]

    return df_raw[use]

def get_dis2std(df, cut):
    use_parts = ["Nos","Ley","Rey","Lea","Rea","Nec","Lsh","Rsh","Lel","Rel",]
    i, status = 0, 0
    len_df = len(df)
    #standard divation for distance of each parts (10 dimensions)
    data = np.zeros((1,10))

    while i <= len_df:
        #cut data to get distance
        if i + cut <= len_df:
            df_tmp = df[i:i+cut]
        #remove remained data after cut
        else:
            break

        dis_points = np.array([])
        #calculate distance for each points(10 points)
        for idx in range(10):
            #X,Y df for each
            df_XY = df_tmp.iloc[:, idx*2:idx*2+2]
            cutmv = np.array(df_XY[(df_XY != 0).all(1)])
            
            dis = np.array([-1])
            if len(cutmv) > 1:
                #get average position for each parts
                avg_pos = np.average(cutmv, axis=0)

                #get distance (L2 norm)
                for pos_xy in cutmv:
                    dis_tmp = np.sqrt(np.sum(np.square(pos_xy-avg_pos)))
                    dis = np.append(dis, dis_tmp)
                
                #get standard deviation
                dis = np.std(dis)
                
            #dis==-1 => no data in cut range
            dis_points = np.append(dis_points, dis)
        
        #process -1
        mv_idx = np.where(dis_points==-1)[0]
        if len(mv_idx) > 0:
            dis_points[np.where(dis_points==-1)[0]] = np.average(dis_points[np.where(dis_points!=-1)[0]])
        data = np.concatenate((data, dis_points.reshape(1,-1)), axis=0)
        
        i += cut

    df_prepared = pd.DataFrame(data[1:], columns=use_parts)
    return df_prepared


##main(std)

#case1: preprocessing for file
if args.file:
    df_prepared = file_preprocessing(args.file, args.cut)

    #save to pickle
    df_prepared.to_pickle('../0-data/data_prepared/'+args.file[:-6]+'_'+str(args.cut)+'.pkl')

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
        print("filename",_)
        df_prepared = file_preprocessing(_, args.cut)
        df_merged = pd.concat([df_merged, df_prepared])

    #shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    #save to pickle
    df_merged.to_pickle('../0-data/data_prepared/merged/'+args.name+'_'+str(args.cut)+'.pkl')
'''
#main(dis2std)
if args.file:
    df_raw = pd.read_pickle('../0-data/data_pickle/'+args.file)
    df_rearrange = rearrange_dis(df_raw)
    df_prepared = get_dis2std(df_rearrange, args.cut)
    
    #add label
    df_prepared['label'] = int(args.file[-5])
    
    #save to pickle
    df_prepared.to_pickle('../0-data/data_prepared/dis2std/'+args.file[:-6]+'_'+str(args.cut)+'.pkl')
'''