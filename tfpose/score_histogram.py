import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

body = ["Nos_Score", "Nec_Score", "Rsh_Score", "Rel_Score", "Rwr_Score", "Lsh_Score", "Lel_Score", "Lwr_Score", "Rey_Score", "Ley_Score", "Rea_Score", "Lea_Score"]

# load dataset
parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

df = pd.read_pickle(args.file)      # x, y, score
df = df[body]

print(df.describe())

# 한 번에 그리기
for i in body:
    part = df[i][df[i] > 0]
    plt.hist(part, bins=100, density=True, alpha=0.5, histtype='step')
plt.savefig('score_histogram.png')

# 12개 따로
'''ax = []
fig = plt.figure(figsize=(40, 30))
for k in range(len(body)):
    ax.append(fig.add_subplot(3, 4, k+1))'''

fig = plt.figure(figsize=(40, 30))
for j in range(len(body)):
    part = df[body[j]][df[body[j]] > 0]
    plt.subplot(3, 4, j+1)
    #ax[j].hist(part, bins=100, density=True, alpha=0.5, histtype='step')
    plt.hist(part, bins=100, density=True, alpha=0.5, histtype='step')
    plt.ylim(0, 30)
    plt.xlabel(body[j])
    plt.ylabel('number')

plt.savefig('score_histogram_4x3.png')