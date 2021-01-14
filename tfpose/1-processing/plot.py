import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


top = ["Nos_X","Nos_Y","Ley_X","Ley_Y","Rey_X","Rey_Y","Lea_X","Lea_Y","Rea_X","Rea_Y"]
mid = ["Nec_X","Nec_Y","Lsh_X","Lsh_Y","Rsh_X","Rsh_Y","Lel_X","Lel_Y","Rel_X","Rel_Y"]

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, default='', help='raw (pickle)data path and name with ".pkl"')
args = parser.parse_args()

# read pickle data
df_raw = pd.read_pickle('../0-data/data_pickle/'+args.file)

# make top, mid data
data_top = df_raw[top].to_numpy()
data_mid = df_raw[mid].to_numpy()

top_len = len(top)
mid_len = len(mid)

# data = np.append(data_top, data_mid)
fig, axes = plt.subplots(nrows=2, ncols=9)

# top = ["Nos_X","Nos_Y","Ley_X","Ley_Y","Rey_X","Rey_Y","Lea_X","Lea_Y","Rea_X","Rea_Y"]
# mid = ["Nec_X","Nec_Y","Lsh_X","Lsh_Y","Rsh_X","Rsh_Y","Lel_X","Lel_Y","Rel_X","Rel_Y"]

# define concat function
def concat_part(data, len):
    data_X = np.array([])
    data_Y = np.array([])
    for i in range(len//2):
        data_X = np.append(data_X, data[:,2*i])
        data_Y = np.append(data_Y, data[:,2*i+1])
    
    return data_X, data_Y

# define draw 2d-histogram function
def drawPart2dHist(data, len, row, part):    
    for i in range (len//2):
        X = data[:,2*i]
        Y = data[:,2*i + 1]
        
        # 마이너스 붙여서 화면에서 처럼 보이게 할지 그냥 0, 1 사이로 할지는 나중에~
        axes[row, i].hist2d(X[np.where(X != 0)], -Y[np.where(Y != 0)], bins=nbins, range=[[0, 1], [-1, 0]])
        axes[row, i].set_xlabel(part[2*i], fontsize=10)
        axes[row, i].set_ylabel(part[2*i+1], fontsize=10)

def drawTotal2dHist(X, Y, len, row, part):
    axes[row, len//2].hist2d(X[np.where(X != 0)],
                            -Y[np.where(Y != 0)],
                            bins=nbins, range=[[0,1], [-1,0]])
    axes[row, len//2].set_xlabel(part + '_X', fontsize=10)
    axes[row, len//2].set_xlabel(part + '_Y', fontsize=10)

# define plot function
def draw1dHist(X, Y, len, row, part):
    axes[row, len//2 + 1].hist(X[np.where(X!=0)], bins=50, range=(0, 1))
    axes[row, len//2 + 1].set_xlabel(part + '_X', fontsize=10)

    axes[row, len//2 + 2].hist(Y[np.where(Y!=0)], bins=50, range=(0, 1))
    axes[row, len//2 + 2].set_xlabel(part + '_Y', fontsize=10)

    tot = np.append(X, Y)
    axes[row, len//2 + 3].hist(tot[np.where(tot!=0)], bins=50, range=(0, 1))
    axes[row, len//2 + 3].set_xlabel(part + '_ALL', fontsize=10)



nbins = 25

# draw each part 2d histogram
drawPart2dHist(data_top, top_len, 0, top)
drawPart2dHist(data_mid, mid_len, 1, mid)

# (10, ) -> (2, )
data_top_X, data_top_Y = concat_part(data_top, top_len)
data_mid_X, data_mid_Y = concat_part(data_mid, mid_len)

# draw top/mid 2d histogram
drawTotal2dHist(data_top_X, data_top_Y, top_len, 0, 'Top')
drawTotal2dHist(data_mid_X, data_mid_Y, top_len, 1, 'Mid')

# draw 1d histogram
draw1dHist(data_top_X, data_top_Y, top_len, 0, 'Top')
draw1dHist(data_mid_X, data_mid_Y, top_len, 1, 'Mid')

plt.show()
