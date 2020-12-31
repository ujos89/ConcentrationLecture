import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--path', type=str, required=True, help='pickle file to read')
args = parser.parse_args()

df = pd.read_pickle(args.path)
print(df)