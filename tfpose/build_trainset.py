import argparse
import os
import pandas as pd

# parser
parser = argparse.ArgumentParser(description='for build train set...')
parser.add_argument('--name', type=str, required=True, help='name for merge')
parser.add_argument('--index', type=int, default=0, help='index to indicate dataframe')
args = parser.parse_args()

def mergebyname(files, name):
    #list to merge
    add_file = []
    for file_ in files:
        if file_.startswith(name):
            add_file.append(file_)

    #merged dataset by name
    df_merged = pd.DataFrame()
    for file_ in add_file:
        tmp_df = pd.read_pickle('data_prepared/'+file_)
        df_merged = pd.concat([df_merged, tmp_df])

    #shuffle
    df_prepared = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_prepared

## save dataframe to pickle
files = os.listdir('./data_prepared')
df_prepared = mergebyname(files, args.name)
df_prepared.to_pickle('train_set/'+args.name+'_'+args.index+'.pkl')