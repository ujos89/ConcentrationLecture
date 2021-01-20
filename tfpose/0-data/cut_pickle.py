import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--path', type=str, required=True, help='pickle file to read')
parser.add_argument('--length', type=int, required=True, help='length of cut frame(min)')
args = parser.parse_args()

df = pd.read_pickle(args.path)
# min to frame
length = args.length*20*60

df_cut = df[:length].reset_index(drop=True)
df_cut.to_pickle('data_prepared/cut/'+args.path[12:-4]+'_'+str(length)+'.pkl')