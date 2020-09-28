import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.functions import *
import sys
import copy

n_train = 300
n_neurons = 150
prop = 9 # prop/10 samples are zero
sparsity = 1-prop/10
data = np.sign(np.random.randint(low=-prop, high=2, size=(n_train, n_neurons)))
data[data < 0] = 0

weight_matrix = np.zeros((data.shape[1], data.shape[1]))

res = []
thetas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
for theta in thetas:
    weight_matrix = train_hebb_zero(data.T, weight_matrix, max_iter=10)
    resa = []
    for i in range(n_train):
        training = copy.deepcopy(data[0:i+1, :])
        training = np.reshape(training, (i+1, n_neurons))
        weight_matrix = train_hebb_zero(training.T, max_iter=1)
        result = np.array([hamming_distance(train, iterate_data_zero(train, weight_matrix, theta, max_iter=50)) for train in training]) # count how many elements
        resa.append(len(result) - np.count_nonzero(result))
    res.append(max(resa))
plt.plot(thetas, res)
plt.xlabel(f'θ')
plt.title(f'Maximal amount of learnable patterns vs bias for sparsity {sparsity:.2f}')
plt.show()



