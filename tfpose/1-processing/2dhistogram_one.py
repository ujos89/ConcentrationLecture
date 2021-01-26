import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import plotly.express as px

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
parser.add_argument('--label', type=int)
args = parser.parse_args()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 25})

histo = pd.DataFrame(columns=['X', 'Y', 'Score'])

df = pd.read_pickle('../0-data/data_pickle/' + args.file)      # x, y, score

body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]

for i in body:
    XY = pd.concat([df[i + '_X'], df[i + '_Y'], df[i + '_Score']], axis=1)
    XY.columns = ['X', 'Y', 'Score']
    histo = histo.append(XY)

histo.Y = - histo.Y

histo = histo[histo.X != 0]
histo = histo[histo.Y != 0]

print(histo)

# fig.show()
nbins=30
plt.figure(figsize=(7, 5))
plt.hist2d(histo['X'], histo['Y'], bins=nbins)
cb = plt.colorbar()
cb.set_label('Number of entries')

ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
#plt.show()

if args.label:
    plt.title("Full-Concentration")
    plt.savefig("kjk_stop")
else:
    plt.title("Non-Concentration")
    plt.savefig("kjk_move")


