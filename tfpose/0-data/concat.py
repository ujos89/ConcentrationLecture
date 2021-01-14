import pandas as pd 
import os
import argparse

parser = argparse.ArgumentParser(description='concat')
parser.add_argument('--file', type=str, required=True, help='name of file')
parser.add_argument('--folder', type=str, required=True, help='name of folder')
parser.add_argument('--label', type=int, required=True, help='label of file')
args = parser.parse_args()
files = os.listdir(args.folder)
add_file = []
for f in files:
    if f.startswith(args.file):
        add_file.append(f)

df_merged = pd.DataFrame()
for f in add_file:
    df = pd.read_pickle(args.folder+'/'+f)
    df_merged = pd.concat([df_merged,df])

df_merged = df_merged.reset_index(drop=True)

df_merged.to_pickle('data_pickle/'+args.file+'_'+str(args.label)+'.pkl')
