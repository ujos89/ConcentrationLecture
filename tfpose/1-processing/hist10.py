import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--c', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
parser.add_argument('--nc', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
args = parser.parse_args()

# read pickle data
df_c = pd.read_pickle('../0-data/data_prepared/'+args.c)        # ( , 5)
df_nc = pd.read_pickle('../0-data/data_prepared/'+args.nc)      # ( , 5)