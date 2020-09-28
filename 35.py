import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys


idx = 2   #9 can be completed!
n_train = 3
print_freq = 50
data = np.reshape(pd.read_csv('data/pict.dat', sep=',', header=None).to_numpy(), (11, 1024))
print(data)

training = data[0:n_train, :].T
weight_matrix = np.zeros((1, 1))
axes = []
fig = plt.figure()

d_range = np.linspace(start=0.0, stop=1, num=10)
weight_matrix = train_hebb(training, weight_matrix=weight_matrix, max_iter=50)
for ind, i in enumerate(d_range):
    noisy = flip_bits(data[idx, :], prop=i)
    iterated = iterate_data(noisy, weight_matrix, max_iter=50)
    axes.append(fig.add_subplot(2, 5, ind+1))
    axes[-1].set_title(f'{ind*10} % bits flipped')
    pix = np.reshape(np.sign(weight_matrix @ iterated), (32, 32))
    plt.imshow(pix.T)
    plt.axis('off')
plt.show()
