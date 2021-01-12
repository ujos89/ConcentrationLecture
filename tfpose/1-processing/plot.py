import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#import preprocessing as pp
import test_pre as pp

# from preprocessing.py
data_top = pp.data_top
data_mid = pp.data_mid

data = np.append(data_top, data_mid)
fig = plt.figure(figsize=(100, 50))

