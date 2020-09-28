import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys

random = False
symmetric = False
idx = 2 #9 can be completed!
data = np.reshape(pd.read_csv('data/pict.dat', sep=',', header=None).to_numpy(), (11, 1024))
training = data[0:3, :].T


if random:
    weight_matrix = np.random.normal(0, 1, (1024, 1024))
elif symmetric:
    rand = np.random.normal(0, 1, (1024, 1))
    weight_matrix = rand + rand.T
    np.fill_diagonal(weight_matrix, 0)
else:
    weight_matrix = np.zeros((1, 1))
    weight_matrix = train_hebb(training, weight_matrix=weight_matrix, max_iter=100)

energy_init = simple_energy(data[idx, :], weight_matrix)
print(energy_init)

_, energy = update_sequential(data[idx, :], weight_matrix, n_iterations=3)
print(energy[-1])
plt.plot(energy)
plt.title(f'Energy of pattern {idx+1}')
plt.show()
