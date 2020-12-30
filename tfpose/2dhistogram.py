import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

parser = argparse.ArgumentParser(description='for preprocessing tfpose data...')
parser.add_argument('--file', type=str, required=True, help='데이터 경로')
args = parser.parse_args()

histo = pd.DataFrame(columns=['X', 'Y', 'Score'])

df = pd.read_pickle(args.file)      # x, y, score

body = ["Nos", "Nec", "Rsh", "Rel", "Rwr", "Lsh", "Lel", "Lwr", "Rey", "Ley", "Rea", "Lea"]

for i in body:
    XY = pd.concat([df[i + '_X'], df[i + '_Y'], df[i + '_Score']], axis=1)
    XY.columns = ['X', 'Y', 'Score']
    histo = histo.append(XY)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(histo['X'], histo['Y'], bins=50)

plt.title('3D histogram of 2D normally distributed data points')
plt.xlabel('x axis')
plt.ylabel('y axis')

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

plt.show()

# python3 2dhistogram.py --file Desktop/tfpose/hci_tfpose/data_pickle/kjk_C03_1.pkl